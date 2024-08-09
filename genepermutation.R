codon.dict <- list(
  "A" = c("GCT", "GCC", "GCA", "GCG"),  # Alanine
  "C" = c("TGT", "TGC"),                # Cysteine
  "D" = c("GAT", "GAC"),                # Aspartic acid
  "E" = c("GAA", "GAG"),                # Glutamic acid
  "F" = c("TTT", "TTC"),                # Phenylalanine
  "G" = c("GGT", "GGC", "GGA", "GGG"),  # Glycine
  "H" = c("CAT", "CAC"),                # Histidine
  "I" = c("ATT", "ATC", "ATA"),         # Isoleucine
  "K" = c("AAA", "AAG"),                # Lysine
  "L" = c("TTA", "TTG", "CTT", "CTC", "CTA", "CTG"),  # Leucine
  "M" = c("ATG"),                       # Methionine
  "N" = c("AAT", "AAC"),                # Asparagine
  "P" = c("CCT", "CCC", "CCA", "CCG"),  # Proline
  "Q" = c("CAA", "CAG"),                # Glutamine
  "R" = c("CGT", "CGC", "CGA", "CGG", "AGA", "AGG"),  # Arginine
  "S" = c("TCT", "TCC", "TCA", "TCG", "AGT", "AGC"),  # Serine
  "T" = c("ACT", "ACC", "ACA", "ACG"),  # Threonine
  "V" = c("GTT", "GTC", "GTA", "GTG"),  # Valine
  "W" = c("TGG"),                       # Tryptophan
  "Y" = c("TAT", "TAC")                 # Tyrosine
)

stop.codons <- list("*" = c("TAA", "TAG", "TGA"))

all.dict <- c(codon.dict, stop.codons)

tokenize_triplets <- function(sequence) {
  sequence <- sequence[1:(length(sequence) - length(sequence) %% 3)]
  triplets <- sapply(seq(1, length(sequence), by = 3), function(i) {
    paste(sequence[i:(i + 2)], collapse = "")
  })
  return(triplets)
}

triplets_keying <- function(triplets, dict=all.dict) {
  keys <- sapply(triplets, function(triplet) {
    for (key in names(dict)) {
      if (triplet %in% dict[[key]]) {
        return(key)
      }
    }
    return("X")
  })
  return(keys)
}

permute_sequence <- function(sequence, type="ok",
                             min.subs, max.subs,
                             dict=codon.dict, spec.cond=FALSE,
                             spec.region=NULL) {
  mutable_sequence <- sequence
  keyed_sequence <- triplets_keying(mutable_sequence, dict)
  num.sub <- sample(min.subs:max.subs, 1)
  if (spec.cond) {
    sub.indices <- sample(spec.region, num.sub)
  } else {
    all_indices <- 1:(length(mutable_sequence) - 1)  # Exclude the last index by default
    eligible_indices <- setdiff(all_indices, spec.region)
    sub.indices <- sample(eligible_indices, num.sub)
      #sample from not spec.region, and not the last one of the vector
      #vorher: sample(1:(length(mutable_sequence)-1), num.sub)
  }
  if (type == "ok") {
    replacements <- sapply(sub.indices, function(i) {
      sample(dict[[keyed_sequence[i]]], 1)
    })
  } else if (type == "func") {
    replacements <- sapply(sub.indices, function(i) {
      sample(unlist(dict), 1)
      # Exclude the current key's values from the list
      #available_elements <- unlist(dict[-which(names(dict) == keyed_sequence[i])])
      #sample(available_elements, 1)
    })
  } else {
    stop("Invalid type")
  }
  for (i in 1:length(sub.indices)) {
    mutable_sequence[sub.indices[i]] <- replacements[i]
  }
  return(mutable_sequence)
}

GenePermutation <- function(sequence, num.perm,
                            min.subs, max.subs,
                            dict=codon.dict,
                            spec.region=NULL) {
  permuted <- data.frame(
    seq = character(),
    label = character()
  )
  for (i in 1:num.perm) {
    label <- sample(c("normal", "abnormal", "special"), size = 1) #, prob = c(0.6, 0.2, 0.2))
    permuted_seq <- permute_sequence(sequence,
                                     type="ok", min.subs=min.subs,
                                     max.subs=max.subs,
                                     dict=dict, spec.cond=FALSE,
                                     spec.region=NULL)
    if (label == "abnormal") {
      permuted_seq <- permute_sequence(permuted_seq,
                                       type="func", min.subs=min.subs,
                                       max.subs=max.subs,
                                       dict=dict, spec.cond=FALSE,
                                       spec.region=spec.region)
    } else if (label == "special") {
      permuted_seq <- permute_sequence(permuted_seq,
                                       type="func", min.subs=min.subs,
                                       max.subs=max.subs,
                                       dict=dict, spec.cond=TRUE,
                                       spec.region=spec.region)
    }
    permuted <- rbind(permuted, data.frame(seq = I(list(permuted_seq)), label = label))
  }
  return(permuted)
}


