
df1 <- read.csv("train_data.csv")
df2 <- read.csv("train_data_no.csv")

library(ggplot2)
library(gridExtra)

in_sample_bench<-206310

out_sample_bench<-252990


# scatterplot for In-sample vs Out-sample performance for each model
p1 <- ggplot(df1, aes(x=In_sample_total_cum, y=Out_sample_total_cum)) +
  geom_point(color = 'blue') +
  labs(x='In-Sample Total Cumulative Value', y='Out-Sample Total Cumulative Value') +
  theme_minimal() +
  ggtitle("Scatterplot: In-Sample vs Out-Sample Performance for Model 1") 

p2 <- ggplot(df2, aes(x=In_sample_total_cum, y=Out_sample_total_cum)) +
  geom_point(color = 'red') +
  labs(x='In-Sample Total Cumulative Value', y='Out-Sample Total Cumulative Value') +
  theme_minimal() +
  ggtitle("Scatterplot: In-Sample vs Out-Sample Performance for Model 2") 

# Boxplots to compare In-sample and Out-sample performance between models
p3 <- ggplot() +
  geom_boxplot(data=df1, aes(x="Model 1", y=In_sample_total_cum), fill = 'blue', alpha = 0.5) +
  geom_boxplot(data=df2, aes(x="Model 2", y=In_sample_total_cum), fill = 'red', alpha = 0.5) +
  labs(x='Model', y='In-Sample Total Cumulative Value') +
  theme_minimal() +
  ggtitle("Boxplot: In-Sample Performance Comparison")

p4 <- ggplot() +
  geom_boxplot(data=df1, aes(x="Model 1", y=Out_sample_total_cum), fill = 'blue', alpha = 0.5) +
  geom_boxplot(data=df2, aes(x="Model 2", y=Out_sample_total_cum), fill = 'red', alpha = 0.5) +
  labs(x='Model', y='Out-Sample Total Cumulative Value') +
  theme_minimal() +
  ggtitle("Boxplot: Out-Sample Performance Comparison")

grid.arrange(p1, p2, p3, p4, ncol = 2)


df1$model <- "Model 1"
df2$model <- "Model 2"

# Combine the dataframes
df <- rbind(df1, df2)

# Melt the data into a long format
library(reshape2)
df <- melt(df, id.vars=c("model"), variable.name="type", value.name="cash")


df <- melt(df, id.vars=c("model"), variable.name="type", value.name="cash")

# Now create the histogram
g1<-ggplot(df, aes(x=cash, fill=model)) +
  geom_histogram(position="identity", alpha=0.5, bins=30) +
  facet_grid(. ~ type) +
  theme_minimal() +
  labs(x="Cash", y="Count", title="Performance comparison of Model 1 and Model 2")

g2<-# Create a density plot
  ggplot(df, aes(x=cash, fill=model)) +
  geom_density(alpha=0.5) +
  facet_grid(. ~ type) +
  theme_minimal() +
  labs(x="Cash", y="Density", title="Density comparison of Model 1 and Model 2")

g3<-# Create a box plot
  ggplot(df, aes(x=model, y=cash, fill=model)) +
  geom_boxplot() +
  facet_grid(. ~ type) +
  theme_minimal() +
  labs(x="Model", y="Cash", title="Boxplot comparison of Model 1 and Model 2")


summary(df1$In_sample_total_cum)
summary(df1$Out_sample_total_cum)
summary(df2$In_sample_total_cum)
summary(df2$Out_sample_total_cum)





# Conduct correlation test to check relationship between In-sample and Out-sample performance
cor.test(df1$In_sample_total_cum, df1$Out_sample_total_cum)
cor.test(df2$In_sample_total_cum, df2$Out_sample_total_cum)

# Perform t-test to compare In-sample and Out-sample performance between models
t.test(df1$In_sample_total_cum, df2$In_sample_total_cum)
t.test(df1$Out_sample_total_cum, df2$Out_sample_total_cum)

# Comparison when In-sample performance beats the benchmark
df1_bench <- df1[df1$In_sample_total_cum > in_sample_bench,]
df2_bench <- df2[df2$In_sample_total_cum > in_sample_bench,]

# t-test to compare Out-sample performance between models when they beat the In-sample benchmark
t.test(df1_bench$Out_sample_total_cum, df2_bench$Out_sample_total_cum)


