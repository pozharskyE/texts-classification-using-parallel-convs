def sample_with_even_distribution_by_col(df, col_name):
  rarest_count = min(df[col_name].value_counts().to_dict().values())
  res_df = pd.DataFrame()
  for val in df[col_name].unique():
    new_chunk = df[df[col_name] == val].sample(rarest_count)
    res_df = pd.concat([res_df, new_chunk])
    
  return res_df

s_df = sample_with_even_distribution_by_col(df, 'overall')