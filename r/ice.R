get_ice_frame <- function(record, col_data, column){
  
  # Add type checking...
  
  is_numeric <- h2o.columns_by_type(col_data, coltype = "numeric")
  is_numeric <- length(is_numeric) > 0
  
  col_data_df <- as.data.frame(col_data)
  
  if(is_numeric){
    # Calculate quantiles if column is numeric
    new_values <- quantile(col_data_df[[column]], probs = seq(0, 1, 0.1))
  } else{
    # Get unique values if column is categorical
    new_values <- unique(col_data_df)
  }
  new_values <- as.h2o(new_values)
  colnames(new_values) <- column
  
  # Create records with range of values
  perturbed_records <- record
  for(i in c(2:nrow(new_values))){
    perturbed_records <- h2o.rbind(perturbed_records, record)
  }
  
  perturbed_records[column] <- NULL
  perturbed_records <- h2o.cbind(perturbed_records, new_values)
  
  return(perturbed_records)
}
