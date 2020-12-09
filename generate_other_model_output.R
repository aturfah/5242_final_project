#!/usr/bin/env Rscript

require(tidyverse)
require(xtable)
require(randomForest)
require(tree)

setwd("~/Documents/projects/5242_final_project/")

### All other models
data <- read_csv("proc_results.csv") %>%
  select(-c("X1", "avg_time")) %>%
  select(-starts_with("train"), -starts_with("valid"), -ends_with("loss")) %>%
  mutate(dataset=factor(dataset, levels=c("MNIST", "Fashion MNIST")),
         architecture=factor(architecture,
                             levels=c("2FC",
                                      "C1-C1-2FC",
                                      "C2-C1-2FC",
                                      "C1-MP-C1-2FC",
                                      "C2-C2-2FC",
                                      "C1-MP-C1-MP-2FC")))

mnist_data <- filter(data, dataset=="MNIST") %>%
  select(-c("dataset"))
fashn_data <- filter(data, dataset=="Fashion MNIST") %>%
  select(-c("dataset"))

performance_summary <- function(data) {
  summarize(data,
            `Mean Test Acc.`=mean(test_acc),
            `Mean Test Acc.`=round(`Mean Test Acc.`, 3),
            `Med. Test Acc.`=median(test_acc),
            `Med. Test Acc.`=round(`Med. Test Acc.`, 3),
            ) %>%
    arrange(desc(`Mean Test Acc.`)) %>%
    xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE)
}


# data %>%
#   ggplot(aes(x=test_acc)) +
#   geom_histogram(fill="white", color="black", binwidth=0.03) +
#   geom_density(alpha=0.8) +
#   labs(title="", x="Test Accuracy", y="Count") +
#   facet_grid(architecture ~ dataset) +
#   theme_classic()

png("images/architecture_comparison.png", height=800, width=800)
data %>%
  ggplot(aes(x=test_acc, fill=architecture)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(architecture ~ dataset) +
  labs(fill="Architecture", linetype="Architecture") +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position="none")
dev.off()

### Just Optimizer
png("images/optimizer_comparison.png", height=500, width=1000)
data %>%
  ggplot(aes(x=test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(optimizer ~ dataset) +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "none")
dev.off()

png("images/regularizer_comparison.png", height=500, width=1000)
data %>%
  ggplot(aes(x=test_acc, fill=regularization)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy", y="Density") +
  facet_grid(regularization ~ dataset) +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "none")
dev.off()


### Optimizer by Regularizer
png("images/optimization_regularization_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Optimizer") +
  facet_grid(regularization ~ dataset) +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()

### Optimizer by Architecture
png("images/optimization_architecture_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=test_acc, fill=optimizer)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Optimizer") +
  facet_grid(architecture ~ dataset) +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()

png("images/regularization_architecture_comparison.png", height=600, width=1000)
data %>%
  ggplot(aes(x=test_acc, fill=regularization)) +
  geom_density(alpha=0.5, size=0.7) +
  scale_fill_hue(h=c(30, 330), l=70, direction=-1) +
  labs(title="", x="Test Accuracy",
       y="Density", fill="Regularization") +
  facet_grid(architecture ~ dataset) +
  theme_classic() +
  theme(text=element_text(size=14),
        legend.position = "right")
dev.off()


########### MNIST ###############

## Best Performance
cat("\n\n\nMNIST Top Performers\n")
mnist_data %>%
  arrange(desc(test_acc)) %>%
  head(8) %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
  print(include.rownames=FALSE)


## Compare results across Architectures
cat("\n\n\nMNIST Architectures\n")
mnist_data %>%
  group_by(architecture) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across Initialization
cat("\n\n\nMNIST Initialization\n")
mnist_data %>%
  group_by(initializer) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across optimizer
cat("\n\n\nMNIST Optimizers\n")
mnist_data %>%
  group_by(optimizer) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across Regularizer
cat("\n\n\nMNIST Regularizers\n")
mnist_data %>%
  group_by(regularization) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)


#### Testing some RF Stuff
random_forest_results <- function(data, out_fname) {
  tree_data <- data %>%
    select(-c("test_loss")) %>%
    mutate(regularization=as.factor(regularization),
           initializer=as.factor(initializer),
           optimizer=as.factor(optimizer))  
  data_rf <- randomForest(test_acc ~ .,
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

random_forest_results(mnist_data, "images/mnist_rf_variable_importance.png")


for (temp in 1:5) {
  cat("#####################################################################\n")
}

########### FASNION MNIST ################
## Best Performance
cat("\n\n\nFASHION_MNIST Top Performers\n")
fashn_data %>%
  arrange(desc(test_acc)) %>%
  head(8) %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE) %>%
  print(include.rownames=FALSE)

## Compare results across Architectures
cat("\n\n\nFASHION_MNIST Architectures\n")
fashn_data %>%
  group_by(architecture) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across Initialization
cat("\n\n\nFASHION_MNIST Initialization\n")
fashn_data %>%
  group_by(initializer) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across optimizer
cat("\n\n\nFASHION_MNIST Optimizers\n")
fashn_data %>%
  group_by(optimizer) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)

## Compare results across Regularizer
cat("\n\n\nFASHION_MNIST Regularizers\n")
fashn_data %>%
  group_by(regularization) %>%
  performance_summary() %>%
  print(include.rownames=FALSE)


random_forest_results(fashn_data, "images/fashn_rf_variable_importance.png")


for (temp in 1:5) {
  cat("#####################################################################\n")
}

cat("\n\n\n\n\nAPPENDIX TABLES (ALL MNIST)\n\n")
mnist_data %>%
  arrange(desc(test_acc)) %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE)


cat("\n\n\n\n\nAPPENDIX TABLES (ALL FASHION_MNIST)\n\n")
fashn_data %>%
  arrange(desc(test_acc)) %>%
  xtable(digits=3, floating=FALSE,latex.environments=NULL,booktabs=TRUE)

