#!/usr/bin/env Rscript

require(tidyverse)
require(xtable)
setwd("~/Documents/projects/5242_final_project/")
### Base Results
base_data <- read_csv("base_results.csv") %>%
  select("dataset", starts_with("train"), starts_with("valid"), starts_with("test")) %>%
  select("dataset", ends_with("acc"))

base_data %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
  print(include.rownames=FALSE)


### Generate training/validaiton convergence graph
dataset_from_filename <- function(fname) {
  if (str_detect(fname, "fashion_mnist")) {
    return("Fashion MNIST")
  } else {
    return("MNIST")
  }
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
           Dataset = factor(Dataset, levels=c("MNIST", "Fashion MNIST")),
           Mode = eval_mode_from_filename(flnm),
           Value_Type = value_type_from_filename(flnm),
           filename = flnm)
}

epoch_data <- list.files(path="base_model_results", pattern = "*csv", full.names=TRUE) %>% 
  map_df(~read_plus(.)) %>%
  group_by(Dataset, Mode, Value_Type, Step) %>%
  summarize(value = mean(Value))

png("images/base_model_convergence.png", height=400, width=800)
epoch_data %>%
  ggplot(aes(x=Step, y=value, color=Mode)) +
  geom_line(alpha=0.5, size=1.5) +
  labs(title="", x="Step", y="Value") +
  facet_grid(Value_Type ~ Dataset, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=16))
dev.off()