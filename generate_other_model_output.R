#!/usr/bin/env Rscript

require(tidyverse)
require(xtable)
require(randomForest)
require(tree)

setwd("~/Documents/projects/5242_final_project/")

### All other models
DATASETS = c("MNIST",
             "Fashion-MNIST",
             "Kuzushiji-MNIST",
             # "CIFAR-10",
             "Kuzushiji-49")
data <- read_csv("proc_results.csv") %>%
  select(-c("X1")) %>%
  select(-starts_with("train"), -starts_with("valid"), -ends_with("loss")) %>%
  mutate(#train_acc_range=max_train_acc - min_train_acc,
         #valid_acc_range=max_valid_acc - min_valid_acc,
         test_acc_range=max_test_acc - min_test_acc) %>%
  select(-starts_with("med_train"), -starts_with("min_train"), -starts_with("max_train"), -starts_with("mean_train"),
         -starts_with("med_valid"), -starts_with("min_valid"), -starts_with("max_valid"), -starts_with("mean_valid")) %>%
  mutate(dataset=factor(dataset, levels=DATASETS),
         architecture=factor(architecture,
                             levels=c("2FC",
                                      "Model C",
                                      "Strided CNN",
                                      "ConvPool CNN",
                                      "All CNN"))) %>%
  arrange(dataset, architecture, regularization, initializer, optimizer) %>% na.omit()

performance_summary <- function(data) {
  summarize(data,
            Num=n(),
            `Mean Test Acc.`=mean(mean_test_acc),
            # `Mean Test Acc.`=round(`Mean Test Acc.`, 4),
            # `Test Acc. Range`=mean(test_acc_range),
            # `Test Acc. Range`=round(`Test Acc. Range`, 4),
            ) %>%
    arrange(desc(`Mean Test Acc.`)) %>%
    xtable(digits=4, floating=FALSE,latex.environments=NULL,booktabs=TRUE)
}



random_forest_results <- function(data, out_fname) {
  tree_data <- data %>%
    filter(architecture != "2FC") %>%
    select(architecture, regularization, initializer, optimizer,
           mean_test_acc) %>%
    mutate(regularization=as.factor(regularization),
           initializer=as.factor(initializer),
           optimizer=as.factor(optimizer))  
  data_rf <- randomForest(mean_test_acc ~ .,
                          mtry=2, ## HOW TO TUNE THIS
                          data = tree_data,
                          importance = TRUE)
  
  cat("\n\n\nRandom Forest Performance\n")
  print(data_rf)
  
  data_imp <- as_tibble(importance(data_rf)) %>%
    mutate(variable=str_to_title(rownames(importance(data_rf)))) %>%
    arrange(desc(`%IncMSE`))
  
  cat("\n\n\nRandom Forest Variable Importance\n")
  data_imp %>%
    rename(Variable=variable,
           `Pct. Inc. MSE`=`%IncMSE`,
           `Inc. Node Purity`=`IncNodePurity`) %>%
    select(Variable, `Pct. Inc. MSE`, `Inc. Node Purity`) %>%
    xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
    print(include.rownames=FALSE)
  
  png(out_fname, height=300, width=500)
  dotchart(data_imp$`%IncMSE`,
           labels=data_imp$variable,
           xlab="Pct. Inc. MSE")
  dev.off() # Negative values -> Random perturbation does better
}


##### BEGIN PLOTS #####
png("images/architecture_comparison.png", height=800, width=800)
data %>%
  ggplot(aes(x=max_test_acc, fill=architecture)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(dataset ~ architecture, scales="free_y") +
  labs(fill="Architecture", linetype="Architecture") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position="none")
dev.off()

### Just Optimizer
png("images/optimizer_comparison.png", height=500, width=1000)
data %>%
  ggplot(aes(x=max_test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(dataset ~ optimizer, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "none")
dev.off()

png("images/regularizer_comparison.png", height=500, width=1000)
data %>%
  ggplot(aes(x=max_test_acc, fill=regularization)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(dataset ~ regularization, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "none")
dev.off()


### Optimizer by Regularizer
png("images/optimization_regularization_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=max_test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Optimizer") +
  facet_grid(dataset ~ regularization, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()

### Optimizer by Architecture
png("images/optimization_architecture_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=max_test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Optimizer") +
  facet_grid(dataset ~ architecture, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()

png("images/regularization_architecture_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=max_test_acc, fill=regularization)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Regularization") +
  facet_grid(dataset ~ architecture, scales="free_y") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()

##### END PLOTS #####


########### MNIST ###############

for (dataset_name in DATASETS) {
  filt_data <- filter(data, dataset==dataset_name) %>%
    select(-c("dataset"))
  
  ## Best Performance
  cat(paste("\n\n\n", dataset_name, " Top Performers\n", sep=""))
  filt_data  %>%
    arrange(desc(mean_test_acc)) %>%
    select("architecture", "regularization", "initializer", "optimizer", "mean_test_acc") %>%
    head(8) %>%
    xtable(digits=4, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
    print(include.rownames=FALSE)

  ## Compare results across Architectures
  cat(paste("\n\n\n", dataset_name, " Architectures\n", sep=""))
  filt_data %>%
    group_by(architecture) %>%
    performance_summary() %>%
    print(include.rownames=FALSE)

  ## Compare results across Initialization
  cat(paste("\n\n\n", dataset_name, " Initialization\n", sep=""))
  filt_data %>%
    group_by(initializer) %>%
    performance_summary() %>%
    print(include.rownames=FALSE)

  ## Compare results across optimizer
  cat(paste("\n\n\n", dataset_name, " Optimizers\n", sep=""))
  filt_data %>%
    group_by(optimizer) %>%
    performance_summary() %>%
    print(include.rownames=FALSE)

  ## Compare results across Regularizer
  cat(paste("\n\n\n", dataset_name, " Regularizers\n", sep=""))
  filt_data %>%
    group_by(regularization) %>%
    performance_summary() %>%
    print(include.rownames=FALSE)
  
  ## RF. Analysis
  output_fname = paste("images/", dataset_name,
                       "_rf_variable_importance.png", sep="")
  random_forest_results(filt_data, output_fname)
}
rm(list=c("filt_data", "output_fname", "dataset_name"))

