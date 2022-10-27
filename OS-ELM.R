matrix_vector<-function(number){
  the_vector<-matrix(seq(1,by=0,length=number),ncol=1,nrow=number)
  return(the_vector)
}

sigmoid_function<-function(x){
  x<-as.matrix(x)
  x_nrow<-nrow(x)
  x_ncol<-ncol(x)
  y<-matrix(nrow=x_nrow,ncol=x_ncol)
  for(i in 1:x_nrow){
    for(j in 1:x_ncol){
      y[i,j]<-1/(1+exp(-x[i,j]))
    }
  }
  return(y)
}

obtain_label_matrix = function(labels, num_classes){
  label_matrix = matrix(0, nrow=length(labels), ncol=num_classes)
  for(i in 1:num_classes){
    label_matrix[which(labels==i), i] = 1
  }
  return(label_matrix)
}

normial<-function(x){
  return((2*(x-min(x))/(max(x)-min(x)))-1)
}

obtained_acc_G_mean<-function(x){
  the_sum<-0
  the_G_mean<-1
  for(i in 1:nrow(x)){
    the_sum<-the_sum+x[i,i]
    the_G_mean<-the_G_mean*(x[i,i]/sum(x[i,]))
  }
  the_acc<-the_sum/sum(x)
  the_G_mean<-the_G_mean^(1/nrow(x))
  return(list(the_acc*100,the_G_mean*100))
}

model = function(number_node, num_classes, train_path, samples_number = 0, test_path = "src", single = FALSE, batch_upper = 100) {
  if (single) {
    total_data = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
  } else {
    data_train = read.table(train_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    data_test = read.table(test_path, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    total_data = rbind(data_train, data_test)
    samples_number = nrow(data_train)
  }
  variables_number = ncol(total_data) - 1
  total_data$label = as.numeric(total_data$label)
  total_data$label = as.factor(total_data$label)
  total_data_normial = as.data.frame(lapply(total_data[, c(1:variables_number)], normial))
  total_data = cbind(total_data_normial, total_data[variables_number + 1])
  data = total_data[c(1:samples_number),]
  testing_data = total_data[-c(1:samples_number),]
  categories = seq(1, by = 1, length = num_classes)
  data_variables = as.matrix(data[, c(1:variables_number)])
  instances_labels = data$label
  data_labels = as.data.frame(matrix(seq(0, by = 0, length = nrow(data) * num_classes), ncol = num_classes))
  names(data_labels) = categories
  for (i in 1:num_classes) {
    position = which(instances_labels == i)
    data_labels[position, i] = 1
  }
  data_labels = as.matrix(data_labels)
  input_weight = matrix(rnorm(variables_number*number_node, mean = 0, sd = 1), nrow = variables_number, ncol = number_node)
  input_bisa = matrix(runif(number_node, min = -1, max = 1), nrow = 1, ncol = number_node)
  start_number = 1
  end_number = number_node + 100
  training_data_variables = data_variables[c(start_number:end_number), ]
  training_data_labels = data_labels[c(start_number:end_number), ]
  H = sigmoid_function(training_data_variables%*%input_weight+matrix_vector(nrow(training_data_variables))%*%input_bisa)
  K = t(H)%*%H
  Beta = solve(K)%*%t(H)%*%training_data_labels
  while (end_number < samples_number) {
    start_number = end_number + 1
    end_number = end_number + sample(1:batch_upper, 1)
    if(end_number > samples_number){
      end_number = samples_number
    }
    training_data_variables = data_variables[c(start_number:end_number), ]
    training_data_labels = data_labels[c(start_number:end_number), ]
    H = sigmoid_function(training_data_variables%*%input_weight+matrix_vector(nrow(training_data_variables))%*%input_bisa)
    K = K + t(H) %*% H
    Beta = Beta + solve(K) %*% t(H) %*% (training_data_labels - H %*% Beta)
  }
  testing_data_variables = as.matrix(testing_data[ ,c(1:variables_number)])
  H = sigmoid_function(testing_data_variables %*% input_weight + matrix_vector(nrow(testing_data)) %*% input_bisa)
  aim_result = as.data.frame(H %*% Beta)
  aim_result$result = 0
  for(i in 1:nrow(aim_result)){
    aim_result[i, num_classes + 1] = which.max(aim_result[i, c(1:num_classes)])
  }
  table0 = table(testing_data$label, factor(aim_result$result, ordered = TRUE, levels = seq(1, by = 1, length = num_classes)))
  final_result = obtained_acc_G_mean(table0)
  Acc = final_result[[1]]
  Gmean = final_result[[2]]
  comp = data.frame(Acc, Gmean)
  names(comp) = c("Acc", "Gmean")
  print(comp)
  saver = read.table("D:/Documents/program/data.csv", header = TRUE, sep = ",")
  saver = rbind(saver, comp)
  write.csv(saver, "D:/Documents/program/data.csv", row.names = FALSE)
}

for (number in 1:50) {
  model(100, 5, "D:/Documents/program/pageblocks_train.csv", 3000, "D:/Documents/program/pageblocks_test.csv")
}
