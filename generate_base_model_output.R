#!/usr/bin/env Rscript

require(tidyverse)
require(xtable)
setwd("~/Documents/projects/5242_final_project/")
### Base Results
base_data <- read_csv("base_results.csv") %>%
  select("dataset", ends_with("acc"))

base_data %>%
  group_by(dataset) %>%
  summarize(`Train Acc.`=mean(train_acc),
            `Valid Acc.`=mean(valid_acc),
            `Test Acc.`=mean(test_acc)) %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
  print(include.rownames=FALSE)


### Generate training/validaiton convergence graph
DATASETS = c("MNIST", "Fashion-MNIST", "Kuzushiji-MNIST",
             "CIFAR-10", "Kuzushiji-49")
dataset_from_filename <- function(fname) {
  idx = 0
  if (str_detect(fname, "fashion_mnist")) {
    idx = 2
  } else if (str_detect(fname, "k49")) {
    idx = 5
  } else if (str_detect(fname, "kmnist")) {
    idx = 3
  } else if (str_detect(fname, "cifar10")) {
    idx = 4
  } else if (str_detect(fname, "mnist")) {
    idx = 1
  } else {
    stop("Invalid Dataset")
  }
  return(DATASETS[idx])
}

eval_mode_from_filename <- function(fname) {
  if (str_detect(fname, "train")) {
    return("Train")
  } else {
    return("Validation")
  }
}

value_type_from_filename <- function (fname) {
  if (str_detect(fname, "sparse_categorical_accuracy")) {
    return("Accuracy")
  } else {
    return("Loss")
  }
}

read_plus <- function(flnm) {
  read_csv(flnm) %>% 
    mutate(Dataset = dataset_from_filename(flnm),
           Dataset = factor(Dataset, levels=DATASETS),
           Mode = eval_mode_from_filename(flnm),
           Value_Type = value_type_from_filename(flnm),
           filename = flnm)
}

epoch_data <- list.files(path="base_csv", pattern = "*csv", full.names=TRUE) %>% 
  map_df(~read_plus(.)) %>%
  group_by(Dataset, Mode, Value_Type, Step) %>%
  summarize(value = mean(Value), max_val=max(Value), min_val=min(Value))

png("images/base_model_convergence.png", height=400, width=800)
epoch_data %>%
  ggplot(aes(x=Step, y=value, color=Mode)) +
  geom_line(alpha=0.9, size=0.8) +
  # geom_line(aes(y=min_val), alpha=0.9, size=0.8, linetype='dashed') +
  # geom_line(aes(y=max_val), alpha=0.9, size=0.8, linetype='dashed') +
  labs(title="", x="Step", y="Value") +
  facet_grid(Value_Type ~ Dataset, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=16))
dev.off()