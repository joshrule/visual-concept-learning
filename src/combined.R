library(tidyverse)
library(dplyr)
library(furrr)
library(latex2exp)
library(lme4)
library(lmerTest)
library(orgutils)
library(readr)
library(stringr)
library(tidyboot)

## run maps in parallel
plan(multiprocess)

## This section reads relevant data into memory

import_results <- function() {
    read_csv("../evaluation/v0_2/final_results.csv") %>%
      ## create a few useful columns
      rename(feature_set = featureSet,
             category = class,
             n_training = nTrainingExamples) %>%
      mutate(feature_set = sub("googlenet-binary-(.+)-evaluation.csv",
                                      "\\1",
                                      feature_set),
             n_training_log = log2(n_training),
             n_train_factor = as.factor(n_training)) %>%
      filter(feature_set %in% c("categorical2", "combined2", "general")) %>%
      mutate(feature_set = factor(feature_set)) %>%
      select(feature_set, category, n_training, n_training_log, n_train_factor, split, everything())
}

import_data <- function() {
    read_csv("../evaluation/v0_2/googlenet-combined-results.csv") %>%
      ## create a few useful columns
      mutate(feature_set = factor(feature_set),
             n_training_log = log2(n_training),
             n_train_factor = as.factor(n_training))
}

## This section plots mean performance on various stats by nTrainingExamples and featureSet with 95% CIs (based on SE)

vmr <- function(xs) { var(xs)/mean(xs) }

compute_weight_vmrs <- function(data) {
    data %>% 
      group_by(feature_set, feature_id, category, n_training_log) %>%
      dplyr::summarize(vmr = vmr(weight))
}

compute_mean_weight_vmrs <- function(weight_vmrs) {
    weight_vmrs %>%
      group_by(feature_set, n_training_log) %>%
      dplyr::summarize(
        mean = mean(vmr),
        var = var(vmr),
        n = n(),
        se = var/sqrt(n),
        ci_lower = mean - 1.96 * se,
        ci_upper = mean + 1.96 * se)
}

compute_weight_variances <- function(data) {
    data %>% 
      group_by(feature_set, feature_id, category, n_training_log) %>%
      dplyr::summarize(s2 = var(weight))
}

compute_mean_weight_variances <- function(weight_variances) {
    weight_variances %>%
      group_by(feature_set, n_training_log) %>%
      dplyr::summarize(
        mean = mean(s2),
        var = var(s2),
        n = n(),
        se = var/sqrt(n),
        ci_lower = mean - 1.96 * se,
        ci_upper = mean + 1.96 * se)
}

plot_mean_weight_stats <- function(mean_weight_stats, stat) {
    ## mean performance by nTrainingExamples and featureSet with 95% CIs (based on SE)
    mean_weight_stats <- mean_weight_stats %>% filter(n_training_log > 0)
    ggplot(mean_weight_stats, aes(x = n_training_log, y = mean, color = feature_set)) +
      ##geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
      ##              color = "#666666",
      ##              width = 0.15) +
      geom_line(size = 0.75) +
      xlab(TeX("Log$_2$ Number of Positive Training Examples")) +
      ylab(TeX(paste("Mean", stat))) +
      theme(strip.background = element_blank(),
            strip.text.x = element_blank(),
            legend.background = element_blank(),
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.text.align = 0,
            axis.title = element_text(size = 14)) +
      scale_color_discrete("Feature Set",
                           labels = c(expression(Categorical),
                           expression(Combined),
                           expression(Generic[1]))) +
      ggsave(paste0("mean_weight_", tolower(stat), "s.png"))
}

## This section plots correlations between weight vectors within each feature set and a best known vector

compute_bk_correlations <- function(data, results) {
    trimmed_results <- results %>%
      select(feature_set, category, n_training, split, dprime)
    data %>%
      left_join(trimmed_results) %>%
      ## split by feature_set here: # features differs by feature set
      group_by(feature_set) %>%
      group_split() %>%
      map(bk_correlations_helper) %>%
      bind_rows()
}

bk_correlations_helper <- function (df) {
    vecs <- df %>%
      spread(feature_id, weight, sep= "_") %>%
      select(-split, -n_training_log, -n_train_factor)
    best <- vecs %>%
      group_by(feature_set, category) %>%
      top_n(1,dprime) %>%
      select(-dprime) %>%
      group_by(feature_set, category, n_training) %>%
      group_nest(.key = "best_vec")
    vecs %>%
      select(-dprime) %>%
      group_by(feature_set, category, n_training) %>%
      group_nest(.key = "raw_data") %>%
      left_join(best %>% select(-n_training), by=c("feature_set", "category")) %>%
      mutate(cors = map2(raw_data, best_vec, ~ matrix(cor(t(.x), y=t(.y), method="pearson"), ncol = 1))) %>%
      select(-raw_data, -best_vec) %>%
      unnest()
}

plot_bk_correlations <- function(bk_correlations) {
    bk_means <- bk_correlations %>%
      group_by(feature_set, n_training) %>%
      dplyr::summarize(
        mean = mean(cors),
        var = var(cors),
        n = n(),
        se = var/sqrt(n),
        ci_lower = mean - 1.96 * se,
        ci_upper = mean + 1.96 * se)
    ggplot(bk_means, aes(x = log2(n_training), y = mean, color = feature_set)) +
      geom_point(data=bk_correlations, aes(x = log2(n_training), y = cors, color = feature_set),
                 shape='.',
                 alpha=0.3,
                 position = position_jitterdodge(dodge.width = 0.7, jitter.width = 0.2)) +
      geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                    position=position_dodge(width=0.7),
                    color = "#666666",
                    width = 0.15) +
      geom_line(size = 0.75, position=position_dodge(width=0.7)) +
      xlab(TeX("Log$_2$ Number of Positive Training Examples")) +
      ylab(TeX("Mean Weight Vector Correlation")) +
      theme(strip.background = element_blank(),
            strip.text.x = element_blank(),
            legend.background = element_blank(),
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.text.align = 0,
            axis.title = element_text(size = 14)) +
      scale_color_discrete("Feature Set",
                           labels = c(expression(Categorical),
                           expression(Combined),
                           expression(Generic[1]))) +
      ggsave("bk_correlation_means.png")
}

## This section computes a correlation between activation sums and top 100 proportions
correlate_sums_and_tops <- function(data) {
    com_feats <- data %>%
      filter(feature_set == "combined2") %>%
      mutate(from_cat = feature_id < 2000)
    com_feats_100 <- com_feats %>%
      group_by(n_training, category, split) %>%
      top_n(100, weight) %>%
      group_by(n_training, from_cat, category, split) %>%
      dplyr::summarize(n = n()) %>%
      group_by(n_training, category, split) %>%
      mutate(n_prop = n/sum(n))
    com_feats_sums <- com_feats %>%
      group_by(n_training, from_cat, category, split) %>%
      dplyr::summarize(weight = sum(abs(weight))) %>%
      group_by(n_training, category, split) %>%
      mutate(prop = weight/sum(weight))
    com_summary <- left_join(com_feats_100, com_feats_sums)
    print(com_summary)
    the_cor <- cor.test(com_summary$n_prop, y = com_summary$prop, method = "pearson")    
    print(the_cor)
    com_summary
}

## This section plots the mean weights of combined features based on their source

plot_combined_source_means <- function(data) {
    com_feats <- data %>%
      filter(feature_set == "combined2") %>%
      mutate(from_cat = feature_id < 2000)
    com_feats_means <- com_feats %>%
      group_by(n_training, from_cat, category, split) %>%
      dplyr::summarize(weight = sum(abs(weight))) %>%
      group_by(n_training, from_cat) %>%
      dplyr::summarize(
        mean = mean(weight),
        var = var(weight),
        n = n(),
        se = var/sqrt(n),
        ci_lower = mean - 1.96 * se,
        ci_upper = mean + 1.96 * se)

    ggplot(com_feats_means, aes(x = log2(n_training), y = mean, color = from_cat, group = from_cat)) +
      geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), color = "#666666", width = 0.15) +
      geom_line(size = 0.75) +
      xlab(TeX("Log$_2$ Number of Positive Training Examples")) +
      ylab(TeX("Mean Total Weight")) +
      theme(strip.background = element_blank(),
            strip.text.x = element_blank(),
            legend.background = element_blank(),
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.text.align = 0,
            axis.title = element_text(size = 14)) +
      scale_color_discrete("Source Feature Set",
                           labels = c(expression(Generic[1]),
                                      expression(Categorical))) +
      ggsave("combined_source_weight.png")
}

## This section plots correlations between weight vectors within each feature set

compute_weight_correlations <- function(data) {
    data %>%
      group_by(feature_set) %>%
      group_split() %>%
      map(weight_correlation_helper) %>%
      bind_rows()
}

weight_correlation_helper <- function (df) {
    df %>%
      spread(feature_id, weight, sep= "_") %>%
      select(-split, -n_training_log, -n_train_factor) %>%
      group_by(feature_set, category, n_training) %>%
      group_nest(.key = "raw_data") %>%
      mutate(cors = map(raw_data, ~ matrix(cor(t(.x), method="pearson"), ncol = 1))) %>%
      select(-raw_data) %>%
      unnest()
}

compute_weight_correlation_means <- function(weight_correlations) {
    weight_correlation_means <- weight_correlations %>%
      group_by(feature_set, n_training) %>%
      dplyr::summarize(
        mean = mean(cors),
        var = var(cors),
        n = n(),
        se = var/sqrt(n),
        ci_lower = mean - 1.96 * se,
        ci_upper = mean + 1.96 * se)
}

plot_weight_correlations <- function(correlation_means) {
    ggplot(correlation_means, aes(x = log2(n_training), y = mean, color = feature_set)) +
      geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                    color = "#666666",
                    width = 0.15) +
      geom_line(size = 0.75) +
      xlab(TeX("Log$_2$ Number of Positive Training Examples")) +
      ylab(TeX("Mean Weight Vector Correlation")) +
      theme(strip.background = element_blank(),
            strip.text.x = element_blank(),
            legend.background = element_blank(),
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.text.align = 0,
            axis.title = element_text(size = 14)) +
      scale_color_discrete("Feature Set",
                           labels = c(expression(Categorical),
                           expression(Combined),
                           expression(Generic[1]))) +
      ggsave("weight_correlation_means.png")
}

## This section helps plot mean weight vectors for individual category x feature-set pairs.

compute_mean_weights <- function(data) {
    data %>%
      select(-split, -n_training_log, -n_train_factor) %>%
      group_by(feature_set, category, n_training, feature_id) %>%
      dplyr::summarize(mean = mean(weight),
                       var = var(weight),
                       n = n(),
                       se = var/sqrt(n),
                       ci_lower = mean - 1.96*se,
                       ci_upper = mean + 1.96*se) %>%
      group_by(feature_set, category) %>%
      group_nest(.key="mean_data")
}

plot_mean_weights <- function(mean_weights) {
    ## take the mean weights and create a plot for each feature set x category.
    mean_weights %>%
      sample_n(5) %>%
      pwalk(~plot_mean_weights_single(..1,..2,..3))
}

plot_mean_weights_single <- function(feature_set, category, data) {
    ## plot the mean weights
    ggplot(data) +
      geom_bar(aes(x = feature_id, y = mean), stat="identity") +
      geom_errorbar(aes(x = feature_id, ymin = ci_lower, ymax = ci_upper)) +
      facet_wrap(~n_training) +
      xlab(TeX("Feature")) +
      ylab(TeX("Mean Weight")) +
      theme(strip.background = element_blank(),
            legend.background = element_blank(),
            legend.title = element_text(size = 14),
            legend.text = element_text(size = 14),
            legend.text.align = 0,
            axis.title = element_text(size = 14)) +
      ggsave(paste0("weight_means_", feature_set, "_", category, ".png"), width=21, height=21)
}

## This section helps with quantitative analysis.

model_bk_correlations <- function(bk_correlations) {
    model <- lmer(cors ~ feature_set + (1 | category), data = bk_correlations)

    cat("\n# Anova of cors ~ feature_set + (1 | category)\n")
    anv <- as_tibble(anova(model), rownames="term") %>%
      mutate_at(vars(`Sum Sq`, `Mean Sq`, "DenDF", `F value`, `Pr(>F)`), function(x) round(x, 3))
      print(toOrg(anv))

    cat("\n# Drop1 for random effects of cors ~ feature_set + (1 | category)\n")
    tmp <- ranova(model) %>%
      as_tibble(rownames = "term") %>%
      mutate_at(vars(c("logLik", "AIC", "LRT")), function(x) round(x, 3))
      print(toOrg(tmp))

    cat("\n# Drop1 for cors ~ feature_set + (1 | category)\n")
    tmp <- drop1(model, ddf = "lme4", test = "Chi") %>%
      as_tibble(rownames = "term") %>%
      mutate_at(vars(c("AIC", "LRT")), function(x) round(x, 3))
      print(toOrg(tmp))

    cat("\n# Drop1 for cors ~ feature_set\n")
    tmp <- drop1(lm(cors ~ feature_set, bk_correlations),
                 ddf = "lme4",
                 test = "Chi") %>%
      as_tibble(rownames = "term") %>%
      mutate_at(vars(c("Sum of Sq", "RSS", "AIC")), function(x) round(x, 3))
      print(toOrg(tmp))

    model
}

model_data <- function(df) {
    df <- df %>% mutate(n_training_factor = factor(n_training_log))

    model <- lmer(vmr ~ feature_set * n_training_factor + (1 | category), data = df)

    anv <- as_tibble(anova(model), rownames="term") %>%
      mutate_at(vars(`Sum Sq`, `Mean Sq`, "DenDF", `F value`, `Pr(>F)`), function(x) round(x, 3))
      print(toOrg(anv))

    rnv <- as_tibble(ranova(model), rownames="term") %>%
      mutate_at(vars("logLik", "AIC", "LRT"), function(x) round(x, 3))
      print(toOrg(rnv))

    model
}

