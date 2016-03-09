set.seed(169)
obs_df <- mutate(obs_df,
                 .rnorm = rnorm(n=nrow(obs_df))
)

https://eventing.coursera.org/api/redirectStrict/KpJ61Ajb5Z1HoUiy4fIBUk2IHd9dSbHzHNiggqiGEna0feuHfQU6CEREZAPxxiW6hibNNZUZDhFhfKOdH4JSyQ.lSUrTxgPCYkzSsp0nw9TZA.XcZGzk2KRDygTq-z7LfWmZg-M7A03c10z1nlf9whIaU6INz3ujGeFuPK4SpaWslWgmNeXKILKPbsGPYAzW3irLvS4EmPdY1ncBBOInhDcHAqN-97gVG_ohgVdc8MsVt7RHZSr94GY5aTvbLMJxiciLozHICmdCIT3hgjz7FUR-IvuKMj5xyzNaSHCWStR3N_W_wkAl5TNnWuaWOzKgbEUDINV1Bq_Kr31v4jOdCNysrX2OodGHPC_4AiowWAELPvfOQudEKRFSKSyasFJ9i6r3I62PKVRF3HthiouysAhWHQKSmlGjq4k9ULadW2mh7QyFOSynAiQG-UG80B93M9Lec3hhwjeJlBH1AlRRT5luVxhqErKF-fmxGax1s8PWzcfZJfhaSEYH1NlvnUrS-dH92zxdx7thRIX7f9_loA18HAy4hlDvWVIYouRVPLS9JFOFvrbWbiUvnxUviY2p_e_A

require(dplyr)
require(tidyr)
print(head(mdlDf))
tmpDf <- tidyr::gather(mdlDf, 'key', 'value', -stepSize, -l2Penalty)
print(head(tmpDf))
