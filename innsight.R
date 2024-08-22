library(innsight)
par(mfrow=c(2,3))
conv <- convert(model)
mat <- array(onehot_instance, dim = c(1, 4, 450))
deeplift <- run_deeplift(conv, mat, x_ref = mat*0+0.25)
plot_global(deeplift, output_idx = 2)
intgrad <- run_intgrad(conv, mat, x_ref = mat*0+0.25, n=400,  ignore_last_act = T)
plot_global(intgrad, output_idx = 2)

smoothgrad <- run_smoothgrad(conv, mat,
               times_input = TRUE,
               ignore_last_act = FALSE)
plot_global(smoothgrad, output_idx = 2)
expgrad <- run_expgrad(conv, mat,
                       n = 50)
plot_global(expgrad, output_idx = 2)
deepshap <- run_deepshap(conv, mat, data_ref = mat*0+0.25)
plot_global(deepshap, output_idx = 2)

connectionweights <- run_cw(conv, mat)
plot(connectionweights, output_idx = 2)

dlres <- get_result(deeplift)
dlres <- dlres[1,,,2]
rownames(dlres) <- c("A", "C", "G", "T")
dlres <- as.matrix(dlres)
ggseqlogo(dlres, method='custom', seq_type='dna') + xlim(motif_pos-20,motif_end+20) + labs(x="bp", y="DeepLift") + geom_rect(aes(xmin = motif_pos, xmax = motif_end, ymin = -Inf, ymax = Inf), fill = "lightblue", alpha = 0.2) +
  # add motif as text
  geom_text(aes(x = motif_pos + 15, y = 0.2, label = motif), size = 3, color = "blue")

igres <- get_result(intgrad)
igres <- igres[1,,,2]
rownames(igres) <- c("A", "C", "G", "T")
igres <- as.matrix(igres)
ggseqlogo(igres, method='custom', seq_type='dna') + xlim(motif_pos-20,motif_end+20) + labs(x="bp", y="IG") + geom_rect(aes(xmin = motif_pos, xmax = motif_end, ymin = -Inf, ymax = Inf), fill = "lightblue", alpha = 0.2) +
  # add motif as text
  geom_text(aes(x = motif_pos + 15, y = 0.2, label = motif), size = 3, color = "blue")







igori <- integrated_gradients(m_steps = 400,
                    input_seq = onehot_instance,
                    baseline_type = "shuffle",
                    target_class_idx = 2,
                    model = model,
                    num_baseline_repeats = 50)
igmat <- as.data.frame(t(as.matrix(igori)))
rownames(igmat) <- c("A", "C", "G", "T")
igmat <- as.matrix(igmat)
ggseqlogo(igmat, method='custom', seq_type='dna') + xlim(motif_pos-20,motif_end+20) + labs(x="bp", y="IG") + geom_rect(aes(xmin = motif_pos, xmax = motif_end, ymin = -Inf, ymax = Inf), fill = "lightblue", alpha = 0.2) +
  # add motif as text
  geom_text(aes(x = motif_pos + 15, y = 0.02, label = motif), size = 3.25, color = "blue")
sum <- rowSums(as.array(igori))
abs_sum <- rowSums(abs(as.array(igori)))
df25 <- data.frame(abs_sum = abs_sum, sum=sum, position = 1 : 450)

ggplot(df25, aes(x = position, y = abs_sum))+ geom_rect(aes(xmin = motif_pos, xmax = motif_end, ymin = -Inf, ymax = Inf), fill = "lightblue", alpha = 0.2) + geom_point() + theme_bw() + labs(subtitle = "Baseline 0.25")
