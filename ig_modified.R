# modification of integrated gradients

interpolate_seq_m <- function(m_steps = 50,
                            baseline_type = "shuffle",
                            input_seq) {
  if (is.list(input_seq)) {
    baseline <- list()
    for (i in 1:length(input_seq)) {
      input_dim <- dim(input_seq[[i]])
      if (baseline_type == "zero") {
        baseline[[i]] <- array(rep(0, prod(input_dim)), dim = input_dim)
      } else {
        input_dim <- dim(input_seq[[i]])
        baseline[[i]] <- array(input_seq[[i]][ , sample(input_dim[2]), ], dim = input_dim)
      }
    }
  } else {
    if (baseline_type == "zero") {
      baseline <- array(rep(0, prod(dim(input_seq))), dim = dim(input_seq))
    } else {
      baseline <- array(input_seq[ , sample(dim(input_seq)[2]), ], dim = dim(input_seq))
    }
  }
}

compute_gradients <- function(input_tensor, target_class_idx, model, input_idx = NULL, pred_stepwise = FALSE) {
  
  # if (is.list(input_tensor)) {
  #   stop("Stepwise predictions only supported for single input layer yet")
  # }
  
  py_run_string("import tensorflow as tf")
  py$input_tensor <- input_tensor
  py$input_idx <- as.integer(input_idx - 1)
  py$target_class_idx <- as.integer(target_class_idx - 1)
  py$model <- model
  
  if (!is.null(input_idx)) {
    py_run_string(
      "with tf.GradientTape() as tape:
             tape.watch(input_tensor[input_idx])
             probs = model(input_tensor)[:, target_class_idx]
    ")
  } else {
    py_run_string(
      "with tf.GradientTape() as tape:
             tape.watch(input_tensor)
             probs = model(input_tensor)[:, target_class_idx]
    ")
  }
  
  grad <- py$tape$gradient(py$probs, py$input_tensor)
  if (!is.null(input_idx)) {
    return(grad[[input_idx]])
  } else {
    return(grad)
  }
}

integral_approximation <- function(gradients) {
  py_run_string("import tensorflow as tf")
  py$gradients <- gradients
  # riemann_trapezoidal
  py_run_string("
grads_mid = (gradients[:, :-1, :] + gradients[:, 1:, :]) / tf.constant(2.0)
grads_with_endpoints = tf.concat([grads_mid, gradients[:, -1:, :]], axis=1)
")
  py_run_string("integrated_gradients = tf.math.reduce_mean(grads_with_endpoints, axis=0)")
  return(py$integrated_gradients)
}

ig_modified <- function (m_steps = 50, baseline_type = "zero", baseline_onehot = NULL,
                         input_seq, target_class_idx, 
          model, pred_stepwise = FALSE, num_baseline_repeats = 1) {
  library(reticulate)
  library(deepG)
  py_run_string("import tensorflow as tf")
  input_idx <- NULL
  if (num_baseline_repeats > 1 & baseline_type == "zero") {
    warning("Ignoring num_baseline_repeats if baseline is of type \"zero\". Did you mean to use baseline_type = \"shuffle\"?")
  }
  if (num_baseline_repeats == 1 | baseline_type == "zero") {
    if (baseline_type == "modify" & !is.null(baseline_onehot)) {
      # we start modifying the baseline
      # we ask the personalized baseline to be one-hot encoded.
      baseline_seq <- array(baseline_onehot, dim=dim(input_seq))
    } else {
      baseline_seq <- interpolate_seq_m(m_steps = m_steps, baseline_type = baseline_type, 
                                        input_seq = input_seq)
    }
    if (is.list(baseline_seq)) {
      for (i in 1:length(baseline_seq)) {
        baseline_seq[[i]] <- tensorflow::tf$cast(baseline_seq[[i]], 
                                                 dtype = "float32")
      }
    }
    else {
      baseline_seq <- tensorflow::tf$cast(baseline_seq, 
                                          dtype = "float32")
    }
    if (is.list(input_seq)) {
      path_gradients <- list()
      avg_grads <- list()
      ig <- list()
      if (pred_stepwise) {
        path_gradients <- gradients_stepwise(model = model, 
                                             baseline_seq = baseline_seq, target_class_idx = target_class_idx)
      }
      else {
        path_gradients <- compute_gradients(model = model, 
                                            input_tensor = baseline_seq, target_class_idx = target_class_idx, 
                                            input_idx = NULL, pred_stepwise = pred_stepwise)
      }
      for (i in 1:length(input_seq)) {
        avg_grads[[i]] <- integral_approximation(gradients = path_gradients[[i]])
        ig[[i]] <- ((input_seq[[i]] - baseline_seq[[i]][1, 
                                                        , ]) * avg_grads[[i]])[1, , ]
      }
    }
    else {
      if (pred_stepwise) {
        path_gradients <- gradients_stepwise(model = model, 
                                             baseline_seq = baseline_seq, target_class_idx = target_class_idx, 
                                             input_idx = NULL)
      }
      else {
        path_gradients <- compute_gradients(model = model, 
                                            input_tensor = baseline_seq, target_class_idx = target_class_idx, 
                                            input_idx = NULL, pred_stepwise = pred_stepwise)
      }
      avg_grads <- integral_approximation(gradients = path_gradients)
      ig <- (input_seq[1, , ] - baseline_seq[1, , ]) * avg_grads[ , ]
    }
  } else if (num_baseline_repeats != 1 & baseline_type == "shuffle") {
    ig_list <- list()
    for (i in 1:num_baseline_repeats) {
      ig_list[[i]] <- integrated_gradients(m_steps = m_steps, 
                                           baseline_type = "shuffle", input_seq = input_seq, 
                                           target_class_idx = target_class_idx, model = model, 
                                           pred_stepwise = pred_stepwise, num_baseline_repeats = 1)
    }
    ig_stacked <- tensorflow::tf$stack(ig_list, axis = 0L)
    ig <- tensorflow::tf$reduce_mean(ig_stacked, axis = 0L)
  }
  return(ig)
}
