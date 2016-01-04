# UofWashington:MachineLearning:House Prices:: price regression:: Houses_tmplt
bdanalytics  

**  **    
**Date: (Mon) Jan 04, 2016**    

# Introduction:  

Data: Washington Sate:Kings County:Seattle House Prices 
Source: 
    Training:   home_data.csv  
    New:        <obsNewFileName>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

Summary of key steps & error improvement stats:

### Prediction Accuracy Enhancement Options:
- transform.data chunk:
    - derive features from multiple features
    
- manage.missing.data chunk:
    - Not fill missing vars
    - Fill missing numerics with a different algorithm
    - Fill missing chars with data based on clusters 
    
### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- all chunks:
    - at chunk-end rm(!glb_<var>)
    
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- extract.features chunk:
    - Add n-grams for glbFeatsText
        - "RTextTools", "tau", "RWeka", and "textcat" packages
    
- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- fit.all.training chunk:
    - myplot_prediction_classification: displays 'x' instead of '+' when there are no prediction errors 
- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list = ls())
set.seed(12345)
options(stringsAsFactors = FALSE)
source("~/Dropbox/datascience/R/myscript.R")
source("~/Dropbox/datascience/R/mydsutils.R")
```

```
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
source("~/Dropbox/datascience/R/myplclust.R")
source("~/Dropbox/datascience/R/mytm.R")
# Gather all package requirements here
suppressPackageStartupMessages(require(doMC))
glbCores <- 6 # of cores on machine - 2
registerDoMC(glbCores) 

suppressPackageStartupMessages(require(caret))
require(plyr)
```

```
## Loading required package: plyr
```

```r
require(dplyr)
```

```
## Loading required package: dplyr
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
#source("dbgcaret.R")
#packageVersion("snow")
#require(sos); findFn("cosine", maxPages=2, sortby="MaxScore")

# Analysis control global variables
# Inputs
#   url/name = "<pointer>"; sep = choose from c(NULL, "\t")
glbObsTrnFile <- list(name = "home_data.csv") 

glbObsNewFile <- #list(url = "<obsNewFileName>") # default OR
    #list(splitSpecs = list(method = NULL #select from c(NULL, "condition", "sample", "copy")
    #                      ,nRatio = 0.3 # > 0 && < 1 if method == "sample" 
    #                      ,seed = 123 # any integer or glbObsTrnPartitionSeed if method == "sample" 
    #                      )
    #    )                   
    list(splitSpecs = list(method = "sample", nRatio = 0.2, seed = 0))
glbInpMerge <- NULL #: default
#     list(fnames = c("<fname1>", "<fname2>")) # files will be concatenated

glb_is_separate_newobs_dataset <- FALSE    # or TRUE
    glb_split_entity_newobs_datasets <- TRUE  # FALSE not supported - use "copy" for glbObsNewFile$splitSpecs$method # select from c(FALSE, TRUE)
    glb_split_newdata_condition <- NULL # or "is.na(<var>)"; "<var> <condition_operator> <value>"

glbObsDropCondition <- NULL # : default
#   enclose in single-quotes b/c condition might include double qoutes
#       use | & ; NOT || &&    
#   '<condition>' 
    # 'grepl("^First Draft Video:", glbObsAll$Headline)'
#nrow(do.call("subset",list(glbObsAll, parse(text=paste0("!(", glbObsDropCondition, ")")))))
    
glb_obs_repartition_train_condition <- NULL # : default
#    "<condition>" 

glb_max_fitobs <- NULL # or any integer
glbObsTrnPartitionSeed <- 123 # or any integer
                         
glb_is_regression <- TRUE; glb_is_classification <- !glb_is_regression; 
    glb_is_binomial <- NULL # or TRUE or FALSE

glb_rsp_var_raw <- "price"

# for classification, the response variable has to be a factor
glb_rsp_var <- glb_rsp_var_raw # or "price.fctr"

# if the response factor is based on numbers/logicals e.g (0/1 OR TRUE/FALSE vs. "A"/"B"), 
#   or contains spaces (e.g. "Not in Labor Force")
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- NULL 
# function(raw) {
#     return(raw ^ 0.5)
#     return(log(raw))
#     return(log(1 + raw))
#     return(log10(raw)) 
#     return(exp(-raw / 2))
#     ret_vals <- rep_len(NA, length(raw)); ret_vals[!is.na(raw)] <- ifelse(raw[!is.na(raw)] == 1, "Y", "N"); return(relevel(as.factor(ret_vals), ref="N"))
#     as.factor(paste0("B", raw))
#     as.factor(gsub(" ", "\\.", raw))    
#     }

#if glb_rsp_var_raw is numeric:
#print(summary(glbObsAll[, glb_rsp_var_raw]))
#glb_map_rsp_raw_to_var(tst <- c(NA, as.numeric(summary(glbObsAll[, glb_rsp_var_raw])))) 

#if glb_rsp_var_raw is character:
#print(table(glbObsAll[, glb_rsp_var_raw]))
#glb_map_rsp_raw_to_var(tst <- c(NA, names(table(glbObsAll[, glb_rsp_var_raw])))) 

glb_map_rsp_var_to_raw <- NULL 
# function(var) {
#     return(var ^ 2.0)
#     return(exp(var))
#     return(10 ^ var) 
#     return(-log(var) * 2)
#     as.numeric(var)
#     gsub("\\.", " ", levels(var)[as.numeric(var)])
#     c("<=50K", " >50K")[as.numeric(var)]
#     c(FALSE, TRUE)[as.numeric(var)]
# }
# glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(tst))

if ((glb_rsp_var != glb_rsp_var_raw) && is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

# List info gathered for various columns
# <col_name>:   <description>; <notes>

# currently does not handle more than 1 column; consider concatenating multiple columns
# If glbFeatsId == NULL, ".rownames <- as.numeric(row.names())" is the default
glbFeatsId <- "id.date" # choose from c(NULL : default, "<id_feat>") 
glbFeatsCategory <- NULL # choose from c(NULL : default, "<category_feat>")

# User-specified exclusions
glbFeatsExclude <- c(NULL
#   Feats that shd be excluded due to known causation by prediction variable
# , "<feat1", "<feat2>"
#   Feats that are linear combinations (alias in glm)
#   Feature-engineering phase -> start by excluding all features except id & category & work each one in
    ,"id","date","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view"
    ,"condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long"
    ,"sqft_living15","sqft_lot15"
                    ) 
if (glb_rsp_var_raw != glb_rsp_var)
    glbFeatsExclude <- union(glbFeatsExclude, glb_rsp_var_raw)                    

glbFeatsInteractionOnly <- list()
#glbFeatsInteractionOnly[["<child_feat>"]] <- "<parent_feat>"

glb_drop_vars <- c(NULL
                # , "<feat1>", "<feat2>"
                )

glb_map_vars <- NULL # or c("<var1>", "<var2>")
glb_map_urls <- list();
# glb_map_urls[["<var1>"]] <- "<var1.url>"

glb_assign_pairs_lst <- NULL; 
# glb_assign_pairs_lst[["<var1>"]] <- list(from=c(NA),
#                                            to=c("NA.my"))
glb_assign_vars <- names(glb_assign_pairs_lst)

# Derived features; Use this mechanism to cleanse data ??? Cons: Data duplication ???
glbFeatsDerive <- list();

# glbFeatsDerive[["<feat.my.sfx>"]] <- list(
#     mapfn = function(<arg1>, <arg2>) { return(function(<arg1>, <arg2>)) } 
#   , args = c("<arg1>", "<arg2>"))

    # character
#     mapfn = function(Week) { return(substr(Week, 1, 10)) }

#     mapfn = function(descriptor) { return(plyr::revalue(descriptor, c(
#         "ABANDONED BUILDING"  = "OTHER",
#         "**"                  = "**"
#                                           ))) }

#     mapfn = function(description) { mod_raw <- description;
    # This is here because it does not work if it's in txt_map_filename
#         mod_raw <- gsub(paste0(c("\n", "\211", "\235", "\317", "\333"), collapse = "|"), " ", mod_raw)
    # Don't parse for "." because of ".com"; use customized gsub for that text
#         mod_raw <- gsub("(\\w)(!|\\*|,|-|/)(\\w)", "\\1\\2 \\3", mod_raw);
    # Some state acrnoyms need context for separation e.g. 
    #   LA/L.A. could either be "Louisiana" or "LosAngeles"
        # modRaw <- gsub("\\bL\\.A\\.( |,|')", "LosAngeles\\1", modRaw);
    #   OK/O.K. could either be "Oklahoma" or "Okay"
#         modRaw <- gsub("\\bACA OK\\b", "ACA OKay", modRaw); 
#         modRaw <- gsub("\\bNow O\\.K\\.\\b", "Now OKay", modRaw);        
    #   PR/P.R. could either be "PuertoRico" or "Public Relations"        
        # modRaw <- gsub("\\bP\\.R\\. Campaign", "PublicRelations Campaign", modRaw);        
    #   VA/V.A. could either be "Virginia" or "VeteransAdministration"        
        # modRaw <- gsub("\\bthe V\\.A\\.\\:", "the VeteranAffairs:", modRaw);
    #   
    # Custom mods

#         return(mod_raw) }

    # numeric
# Create feature based on record position/id in data   
glbFeatsDerive[[".pos"]] <- list(
    mapfn = function(.rnorm) { return(1:length(.rnorm)) }       
    , args = c(".rnorm"))    

glbFeatsDerive[["id.date"]] <- list(
    mapfn = function(id, date) { return(paste(as.character(id), as.character(date), sep = "#")) }       
    , args = c("id", "date"))    

# Add logs of numerics that are not distributed normally
#   Derive & keep multiple transformations of the same feature, if normality is hard to achieve with just one transformation
#   Right skew: logp1; sqrt; ^ 1/3; logp1(logp1); log10; exp(-<feat>/constant)
# glbFeatsDerive[["WordCount.log1p"]] <- list(
#     mapfn = function(WordCount) { return(log1p(WordCount)) } 
#   , args = c("WordCount"))
# glbFeatsDerive[["WordCount.root2"]] <- list(
#     mapfn = function(WordCount) { return(WordCount ^ (1/2)) } 
#   , args = c("WordCount"))
# glbFeatsDerive[["WordCount.nexp"]] <- list(
#     mapfn = function(WordCount) { return(exp(-WordCount)) } 
#   , args = c("WordCount"))
#print(summary(glbObsAll$WordCount))
#print(summary(mapfn(glbObsAll$WordCount)))
    
#     mapfn = function(HOSPI.COST) { return(cut(HOSPI.COST, 5, breaks = c(0, 100000, 200000, 300000, 900000), labels = NULL)) }     
#     mapfn = function(Rasmussen)  { return(ifelse(sign(Rasmussen) >= 0, 1, 0)) } 
#     mapfn = function(startprice) { return(startprice ^ (1/2)) }       
#     mapfn = function(startprice) { return(log(startprice)) }   
#     mapfn = function(startprice) { return(exp(-startprice / 20)) }
#     mapfn = function(startprice) { return(scale(log(startprice))) }     
#     mapfn = function(startprice) { return(sign(sprice.predict.diff) * (abs(sprice.predict.diff) ^ (1/10))) }        

    # factor      
#     mapfn = function(PropR) { return(as.factor(ifelse(PropR >= 0.5, "Y", "N"))) }
#     mapfn = function(productline, description) { as.factor(gsub(" ", "", productline)) }
#     mapfn = function(purpose) { return(relevel(as.factor(purpose), ref="all_other")) }
#     mapfn = function(raw) { tfr_raw <- as.character(cut(raw, 5)); 
#                             tfr_raw[is.na(tfr_raw)] <- "NA.my";
#                             return(as.factor(tfr_raw)) }
#     mapfn = function(startprice.log10) { return(cut(startprice.log10, 3)) }
#     mapfn = function(startprice.log10) { return(cut(sprice.predict.diff, c(-1000, -100, -10, -1, 0, 1, 10, 100, 1000))) }    

#     , args = c("<arg1>"))
    
    # multiple args    
#     mapfn = function(PTS, oppPTS) { return(PTS - oppPTS) }
#     mapfn = function(startprice.log10.predict, startprice) {
#                  return(spdiff <- (10 ^ startprice.log10.predict) - startprice) } 
#     mapfn = function(productline, description) { as.factor(
#         paste(gsub(" ", "", productline), as.numeric(nchar(description) > 0), sep = "*")) }

# # If glbObsAll is not sorted in the desired manner
#     mapfn=function(Week) { return(coredata(lag(zoo(orderBy(~Week, glbObsAll)$ILI), -2, na.pad=TRUE))) }
#     mapfn=function(ILI) { return(coredata(lag(zoo(ILI), -2, na.pad=TRUE))) }
#     mapfn=function(ILI.2.lag) { return(log(ILI.2.lag)) }

# glbFeatsDerive[["<var1>"]] <- glbFeatsDerive[["<var2>"]]

glb_derive_vars <- names(glbFeatsDerive)

# tst <- "descr.my"; args_lst <- NULL; for (arg in glbFeatsDerive[[tst]]$args) args_lst[[arg]] <- glbObsAll[, arg]; print(head(args_lst[[arg]])); print(head(drv_vals <- do.call(glbFeatsDerive[[tst]]$mapfn, args_lst))); 
# print(which_ix <- which(args_lst[[arg]] == 0.75)); print(drv_vals[which_ix]); 

glbFeatsDateTime <- list()
# glbFeatsDateTime[["<DateTimeFeat>"]] <- 
#     c(format = "%Y-%m-%d %H:%M:%S", timezone = "America/New_York", impute.na = TRUE, 
#       last.ctg = TRUE, poly.ctg = TRUE)

glbFeatsPrice <- NULL # or c("<price_var>")

glbFeatsText <- list()
Sys.setlocale("LC_ALL", "C") # For english
```

```
## [1] "C/C/C/C/C/en_US.UTF-8"
```

```r
#glbFeatsText[["<TextFeature>"]] <- list(NULL,
#   ,names = myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL, 
#       <comma-separated-screened-names>
#   ))))
#   ,rareWords = myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL, 
#       <comma-separated-nonSCOWL-words>
#   ))))
#)

# Text Processing Step: custom modifications not present in txt_munge -> use glbFeatsDerive
# Text Processing Step: universal modifications
glb_txt_munge_filenames_pfx <- "<projectId>_mytxt_"

# Text Processing Step: tolower
# Text Processing Step: myreplacePunctuation
# Text Processing Step: removeWords
glb_txt_stop_words <- list()
# Remember to use unstemmed words
if (length(glbFeatsText) > 0) {
    require(tm)
    require(stringr)

    glb_txt_stop_words[["<txt_var>"]] <- sort(myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
        # Remove any words from stopwords            
#         , setdiff(myreplacePunctuation(stopwords("english")), c("<keep_wrd1>", <keep_wrd2>"))
                                
        # Remove salutations
        ,"mr","mrs","dr","Rev"                                

        # Remove misc
        #,"th" # Happy [[:digit::]]+th birthday 

        # Remove terms present in Trn only or New only; search for "Partition post-stem"
        #   ,<comma-separated-terms>        

        # cor.y.train == NA
#         ,unlist(strsplit(paste(c(NULL
#           ,"<comma-separated-terms>"
#         ), collapse=",")

        # freq == 1; keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # chisq.pval high (e.g. == 1); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>

        # nzv.freqRatio high (e.g. >= glbFeatsNzvFreqMax); keep c("<comma-separated-terms-to-keep>")
            # ,<comma-separated-terms>        
                                            )))))
}
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^man", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txtFeat]][, 4866] > 0, c(glb_rsp_var, txtFeat)]

# To identify terms with a specific freq
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], freq == 1)$term), collapse = ",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], freq <= 2)$term), collapse = ",")
#subset(glb_post_stem_words_terms_df_lst[[txtFeat]], term %in% c("zinger"))

# To identify terms with a specific freq & 
#   are not stemmed together later OR is value of color.fctr (e.g. gold)
#paste0(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], (freq == 1) & !(term %in% c("blacked","blemish","blocked","blocks","buying","cables","careful","carefully","changed","changing","chargers","cleanly","cleared","connect","connects","connected","contains","cosmetics","default","defaulting","defective","definitely","describe","described","devices","displays","drop","drops","engravement","excellant","excellently","feels","fix","flawlessly","frame","framing","gentle","gold","guarantee","guarantees","handled","handling","having","install","iphone","iphones","keeped","keeps","known","lights","line","lining","liquid","liquidation","looking","lots","manuals","manufacture","minis","most","mostly","network","networks","noted","opening","operated","performance","performs","person","personalized","photograph","physically","placed","places","powering","pre","previously","products","protection","purchasing","returned","rotate","rotation","running","sales","second","seconds","shipped","shuts","sides","skin","skinned","sticker","storing","thats","theres","touching","unusable","update","updates","upgrade","weeks","wrapped","verified","verify") ))$term), collapse = ",")

#print(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (freq <= 2)))
#glbObsAll[which(terms_mtrx[, 229] > 0), glbFeatsText]

# To identify terms with cor.y == NA
#orderBy(~-freq+term, subset(glb_post_stop_words_terms_df_lst[[txtFeat]], is.na(cor.y)))
#paste(sort(subset(glb_post_stop_words_terms_df_lst[[txtFeat]], is.na(cor.y))[, "term"]), collapse=",")
#orderBy(~-freq+term, subset(glb_post_stem_words_terms_df_lst[[txtFeat]], is.na(cor.y)))

# To identify terms with low cor.y.abs
#head(orderBy(~cor.y.abs+freq+term, subset(glb_post_stem_words_terms_df_lst[[txtFeat]], !is.na(cor.y))), 5)

# To identify terms with high chisq.pval
#subset(glb_post_stem_words_terms_df_lst[[txtFeat]], chisq.pval > 0.99)
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (chisq.pval > 0.99) & (freq <= 10))$term), collapse=",")
#paste0(sort(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (chisq.pval > 0.9))$term), collapse=",")
#head(orderBy(~-chisq.pval+freq+term, glb_post_stem_words_terms_df_lst[[txtFeat]]), 5)
#glbObsAll[glb_post_stem_words_terms_mtrx_lst[[txtFeat]][, 68] > 0, glbFeatsText]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^m", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])

# To identify terms with high nzv.freqRatio
#summary(glb_post_stem_words_terms_df_lst[[txtFeat]]$nzv.freqRatio)
#paste0(sort(setdiff(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], (nzv.freqRatio >= glbFeatsNzvFreqMax) & (freq < 10) & (chisq.pval >= 0.05))$term, c( "128gb","3g","4g","gold","ipad1","ipad3","ipad4","ipadair2","ipadmini2","manufactur","spacegray","sprint","tmobil","verizon","wifion"))), collapse=",")

# To identify obs with a txt term
#tail(orderBy(~-freq+term, glb_post_stop_words_terms_df_lst[[txtFeat]]), 20)
#mydspObs(list(descr.my.contains="non"), cols=c("color", "carrier", "cellular", "storage"))
#grep("ever", dimnames(terms_stop_mtrx)$Terms)
#which(terms_stop_mtrx[, grep("ipad", dimnames(terms_stop_mtrx)$Terms)] > 0)
#glbObsAll[which(terms_stop_mtrx[, grep("16", dimnames(terms_stop_mtrx)$Terms)[1]] > 0), c(glbFeatsCategory, "storage", txtFeat)]

# Text Processing Step: screen for names # Move to glbFeatsText specs section in order of text processing steps
# glbFeatsText[["<txtFeat>"]]$names <- myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
#         # Person names for names screening
#         ,<comma-separated-list>
#         
#         # Company names
#         ,<comma-separated-list>
#                     
#         # Product names
#         ,<comma-separated-list>
#     ))))

# glbFeatsText[["<txtFeat>"]]$rareWords <- myreplacePunctuation(str_to_lower(gsub(" ", "", c(NULL
#         # Words not in SCOWL db
#         ,<comma-separated-list>
#     ))))

# To identify char vectors post glbFeatsTextMap
#grep("six(.*)hour", glb_txt_chr_lst[[txtFeat]], ignore.case = TRUE, value = TRUE)
#grep("[S|s]ix(.*)[H|h]our", glb_txt_chr_lst[[txtFeat]], value = TRUE)

# To identify whether terms shd be synonyms
#orderBy(~term, glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^moder", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ])
# term_row_df <- glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^came$", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ]
# 
# cor(glb_post_stop_words_terms_mtrx_lst[[txtFeat]][glbObsAll$.lcn == "Fit", term_row_df$pos], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")

# To identify which stopped words are "close" to a txt term
#sort(cluster_vars)

# Text Processing Step: stemDocument
# To identify stemmed txt terms
#glb_post_stop_words_terms_df_lst[[txtFeat]][grep("^la$", glb_post_stop_words_terms_df_lst[[txtFeat]]$term), ]
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^con", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
#glbObsAll[which(terms_stem_mtrx[, grep("use", dimnames(terms_stem_mtrx)$Terms)[[1]]] > 0), c(glbFeatsId, "productline", txtFeat)]
#glbObsAll[which(TfIdf_stem_mtrx[, 191] > 0), c(glbFeatsId, glbFeatsCategory, txtFeat)]
#glbObsAll[which(glb_post_stop_words_terms_mtrx_lst[[txtFeat]][, 6165] > 0), c(glbFeatsId, glbFeatsCategory, txtFeat)]
#which(glbObsAll$UniqueID %in% c(11915, 11926, 12198))

# Text Processing Step: mycombineSynonyms
#   To identify which terms are associated with not -> combine "could not" & "couldn't"
#findAssocs(glb_full_DTM_lst[[txtFeat]], "not", 0.05)
#   To identify which synonyms should be combined
#orderBy(~term, glb_post_stem_words_terms_df_lst[[txtFeat]][grep("^c", glb_post_stem_words_terms_df_lst[[txtFeat]]$term), ])
chk_comb_cor <- function(syn_lst) {
#     cor(terms_stem_mtrx[glbObsAll$.src == "Train", grep("^(damag|dent|ding)$", dimnames(terms_stem_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
    print(subset(glb_post_stem_words_terms_df_lst[[txtFeat]], term %in% syn_lst$syns))
    print(subset(get_corpus_terms(tm_map(glbFeatsTextCorpus[[txtFeat]], mycombineSynonyms, list(syn_lst), lazy=FALSE)), term == syn_lst$word))
#     cor(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])], glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
#     cor(rowSums(terms_stop_mtrx[glbObsAll$.src == "Train", grep("^(damage|dent|ding)$", dimnames(terms_stop_mtrx)[[2]])]), glbObsTrn[, glb_rsp_var], use="pairwise.complete.obs")
}
#chk_comb_cor(syn_lst=list(word="cabl",  syns=c("cabl", "cord")))
#chk_comb_cor(syn_lst=list(word="damag",  syns=c("damag", "dent", "ding")))
#chk_comb_cor(syn_lst=list(word="dent",  syns=c("dent", "ding")))
#chk_comb_cor(syn_lst=list(word="use",  syns=c("use", "usag")))

glbFeatsTextSynonyms <- list()
# list parsed to collect glbFeatsText[[<txtFeat>]]$vldTerms
# glbFeatsTextSynonyms[["Hdln.my"]] <- list(NULL
#     # people in places
#     , list(word = "australia", syns = c("australia", "australian"))
#     , list(word = "italy", syns = c("italy", "Italian"))
#     , list(word = "newyork", syns = c("newyork", "newyorker"))    
#     , list(word = "Pakistan", syns = c("Pakistan", "Pakistani"))    
#     , list(word = "peru", syns = c("peru", "peruvian"))
#     , list(word = "qatar", syns = c("qatar", "qatari"))
#     , list(word = "scotland", syns = c("scotland", "scotish"))
#     , list(word = "Shanghai", syns = c("Shanghai", "Shanzhai"))    
#     , list(word = "venezuela", syns = c("venezuela", "venezuelan"))    
# 
#     # companies - needs to be data dependent 
#     #   - e.g. ensure BNP in this experiment/feat always refers to BNPParibas
#         
#     # general synonyms
#     , list(word = "Create", syns = c("Create","Creator")) 
#     , list(word = "cute", syns = c("cute","cutest"))     
#     , list(word = "Disappear", syns = c("Disappear","Fadeout"))     
#     , list(word = "teach", syns = c("teach", "taught"))     
#     , list(word = "theater",  syns = c("theater", "theatre", "theatres")) 
#     , list(word = "understand",  syns = c("understand", "understood"))    
#     , list(word = "weak",  syns = c("weak", "weaken", "weaker", "weakest"))
#     , list(word = "wealth",  syns = c("wealth", "wealthi"))    
#     
#     # custom synonyms (phrases)
#     
#     # custom synonyms (names)
#                                       )
#glbFeatsTextSynonyms[["<txtFeat>"]] <- list(NULL
#     , list(word="<stem1>",  syns=c("<stem1>", "<stem1_2>"))
#                                       )

for (txtFeat in names(glbFeatsTextSynonyms))
    for (entryIx in 1:length(glbFeatsTextSynonyms[[txtFeat]])) {
        glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$word <-
            str_to_lower(glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$word)
        glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$syns <-
            str_to_lower(glbFeatsTextSynonyms[[txtFeat]][[entryIx]]$syns)        
    }        

glbFeatsTextSeed <- 181
# tm options include: check tm::weightSMART 
glb_txt_terms_control <- list( # Gather model performance & run-time stats
                    # weighting = function(x) weightSMART(x, spec = "nnn")
                    # weighting = function(x) weightSMART(x, spec = "lnn")
                    # weighting = function(x) weightSMART(x, spec = "ann")
                    # weighting = function(x) weightSMART(x, spec = "bnn")
                    # weighting = function(x) weightSMART(x, spec = "Lnn")
                    # 
                    weighting = function(x) weightSMART(x, spec = "ltn") # default
                    # weighting = function(x) weightSMART(x, spec = "lpn")                    
                    # 
                    # weighting = function(x) weightSMART(x, spec = "ltc")                    
                    # 
                    # weighting = weightBin 
                    # weighting = weightTf 
                    # weighting = weightTfIdf # : default
                # termFreq selection criteria across obs: tm default: list(global=c(1, Inf))
                    , bounds = list(global = c(1, Inf)) 
                # wordLengths selection criteria: tm default: c(3, Inf)
                    , wordLengths = c(1, Inf) 
                              ) 

glb_txt_cor_var <- glb_rsp_var # : default # or c(<feat>)

# select one from c("union.top.val.cor", "top.cor", "top.val", default: "top.chisq", "sparse")
glbFeatsTextFilter <- "top.chisq" 
glbFeatsTextTermsMax <- rep(10, length(glbFeatsText)) # :default
names(glbFeatsTextTermsMax) <- names(glbFeatsText)

# Text Processing Step: extractAssoc
glbFeatsTextAssocCor <- rep(1, length(glbFeatsText)) # :default 
names(glbFeatsTextAssocCor) <- names(glbFeatsText)

# Remember to use stemmed terms
glb_important_terms <- list()

# Text Processing Step: extractPatterns (ngrams)
glbFeatsTextPatterns <- list()
#glbFeatsTextPatterns[[<txtFeat>>]] <- list()
#glbFeatsTextPatterns[[<txtFeat>>]] <- c(metropolitan.diary.colon = "Metropolitan Diary:")

# Have to set it even if it is not used
# Properties:
#   numrows(glb_feats_df) << numrows(glbObsFit
#   Select terms that appear in at least 0.2 * O(FP/FN(glbObsOOB)) ???
#       numrows(glbObsOOB) = 1.1 * numrows(glbObsNew) ???
glb_sprs_thresholds <- NULL # or c(<txtFeat1> = 0.988, <txtFeat2> = 0.970, <txtFeat3> = 0.970)

glbFctrMaxUniqVals <- 20 # default: 20
glb_impute_na_data <- FALSE # or TRUE
glb_mice_complete.seed <- 144 # or any integer

glb_cluster <- FALSE # : default or TRUE
glb_cluster.seed <- 189 # or any integer
glb_cluster_entropy_var <- NULL # c(glb_rsp_var, as.factor(cut(glb_rsp_var, 3)), default: NULL)
glbFeatsTextClusterVarsExclude <- FALSE # default FALSE

glb_interaction_only_feats <- NULL # : default or c(<parent_feat> = "<child_feat>")

glbFeatsNzvFreqMax <- 19 # 19 : caret default
glbFeatsNzvUniqMin <- 10 # 10 : caret default

glbRFESizes <- list()
#glbRFESizes[["mdlFamily"]] <- c(4, 8, 16, 32, 64, 67, 68, 69) # Accuracy@69/70 = 0.8258

glbObsFitOutliers <- list()
# If outliers.n >= 10; consider concatenation of interaction vars
# glbObsFitOutliers[["<mdlFamily>"]] <- c(NULL
    # is.na(.rstudent)
    # is.na(.dffits)
    # .hatvalues >= 0.99        
    # -38,167,642 < minmax(.rstudent) < 49,649,823    
#     , <comma-separated-<glbFeatsId>>
#                                     )
glbObsTrnOutliers <- list()

# influence.measures: car::outlier; rstudent; dffits; hatvalues; dfbeta; dfbetas
#mdlId <- "RFE.X.glm"; obs_df <- fitobs_df
#mdlId <- "Final.glm"; obs_df <- trnobs_df
#mdlId <- "CSM2.X.glm"; obs_df <- fitobs_df
#print(outliers <- car::outlierTest(glb_models_lst[[mdlId]]$finalModel))
#mdlIdFamily <- paste0(head(unlist(str_split(mdlId, "\\.")), -1), collapse="."); obs_df <- dplyr::filter_(obs_df, interp(~(!(var %in% glbObsFitOutliers[[mdlIdFamily]])), var = as.name(glbFeatsId))); model_diags_df <- cbind(obs_df, data.frame(.rstudent=stats::rstudent(glb_models_lst[[mdlId]]$finalModel)), data.frame(.dffits=stats::dffits(glb_models_lst[[mdlId]]$finalModel)), data.frame(.hatvalues=stats::hatvalues(glb_models_lst[[mdlId]]$finalModel)));print(summary(model_diags_df[, c(".rstudent",".dffits",".hatvalues")])); table(cut(model_diags_df$.hatvalues, breaks=c(0.00, 0.98, 0.99, 1.00)))

#print(subset(model_diags_df, is.na(.rstudent))[, glbFeatsId])
#print(subset(model_diags_df, is.na(.dffits))[, glbFeatsId])
#print(model_diags_df[which.min(model_diags_df$.dffits), ])
#print(subset(model_diags_df, .hatvalues > 0.99)[, glbFeatsId])
#dffits_df <- merge(dffits_df, outliers_df, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#dffits_df <- merge(dffits_df, glbObsFit, by="row.names", all.x=TRUE); row.names(dffits_df) <- dffits_df$Row.names; dffits_df <- subset(dffits_df, select=-Row.names)
#subset(dffits_df, !is.na(.Bonf.p))

#mdlId <- "CSM.X.glm"; vars <- myextract_actual_feats(row.names(orderBy(reformulate(c("-", paste0(mdlId, ".imp"))), myget_feats_imp(glb_models_lst[[mdlId]])))); 
#model_diags_df <- glb_get_predictions(model_diags_df, mdlId, glb_rsp_var)
#obs_ix <- row.names(model_diags_df) %in% names(outliers$rstudent)[1]
#obs_ix <- which(is.na(model_diags_df$.rstudent))
#obs_ix <- which(is.na(model_diags_df$.dffits))
#myplot_parcoord(obs_df=model_diags_df[, c(glbFeatsId, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, paste0(glb_rsp_var, mdlId), vars[1:min(20, length(vars))])], obs_ix=obs_ix, id_var=glbFeatsId, category_var=glbFeatsCategory)

#model_diags_df[row.names(model_diags_df) %in% names(outliers$rstudent)[c(1:2)], ]
#ctgry_diags_df <- model_diags_df[model_diags_df[, glbFeatsCategory] %in% c("Unknown#0"), ]
#myplot_parcoord(obs_df=ctgry_diags_df[, c(glbFeatsId, glbFeatsCategory, ".rstudent", ".dffits", ".hatvalues", glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:20])], obs_ix=row.names(ctgry_diags_df) %in% names(outliers$rstudent)[1], id_var=glbFeatsId, category_var=glbFeatsCategory)
#table(glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), "startprice.log10.cut.fctr"])
#glbObsFit[model_diags_df[, glbFeatsCategory] %in% c("iPad1#1"), c(glbFeatsId, "startprice")]

# No outliers & .dffits == NaN
#myplot_parcoord(obs_df=model_diags_df[, c(glbFeatsId, glbFeatsCategory, glb_rsp_var, "startprice.log10.predict.RFE.X.glmnet", indep_vars[1:10])], obs_ix=seq(1:nrow(model_diags_df))[is.na(model_diags_df$.dffits)], id_var=glbFeatsId, category_var=glbFeatsCategory)

# Modify mdlId to (build & extract) "<FamilyId>#<Fit|Trn>#<caretMethod>#<preProc1.preProc2>#<samplingMethod>"
glb_models_lst <- list(); glb_models_df <- data.frame()
# Regression
if (glb_is_regression) {
    glbMdlMethods <- c(NULL
        # deterministic
            #, "lm", # same as glm
            , "glm", "bayesglm", "glmnet"
            , "rpart"
        # non-deterministic
            , "gbm", "rf" 
        # Unknown
            , "nnet" , "avNNet" # runs 25 models per cv sample for tunelength=5
            , "svmLinear", "svmLinear2"
            , "svmPoly" # runs 75 models per cv sample for tunelength=5
            , "svmRadial" 
            , "earth"
            , "bagEarth" # Takes a long time
        )
} else
# Classification - Add ada (auto feature selection)
    if (glb_is_binomial)
        glbMdlMethods <- c(NULL
        # deterministic                     
            , "bagEarth" # Takes a long time        
            , "glm", "bayesglm", "glmnet"
            , "nnet"
            , "rpart"
        # non-deterministic        
            , "gbm"
            , "avNNet" # runs 25 models per cv sample for tunelength=5      
            , "rf"
        # Unknown
            , "lda", "lda2"
                # svm models crash when predict is called -> internal to kernlab it should call predict without .outcome
            , "svmLinear", "svmLinear2"
            , "svmPoly" # runs 75 models per cv sample for tunelength=5
            , "svmRadial" 
            , "earth"
        ) else
        glbMdlMethods <- c(NULL
        # deterministic
            ,"glmnet"
        # non-deterministic 
            ,"rf"       
        # Unknown
            ,"gbm","rpart"
        )

glbMdlFamilies <- list(); glb_mdl_feats_lst <- list()
# family: Choose from c("RFE.X", "CSM.X", "All.X", "Best.Interact")
#   methods: Choose from c(NULL, <method>, glbMdlMethods) 
#glbMdlFamilies[["RFE.X"]] <- c("glmnet", "glm") # non-NULL vector is mandatory
glbMdlFamilies[["All.X"]] <- c("glmnet", "glm")  # non-NULL vector is mandatory
#glbMdlFamilies[["Best.Interact"]] <- "glmnet" # non-NULL vector is mandatory

# Check if interaction features make RFE better
# glbMdlFamilies[["CSM.X"]] <- setdiff(glbMdlMethods, c("lda", "lda2")) # crashing due to category:.clusterid ??? #c("glmnet", "glm") # non-NULL list is mandatory
# glb_mdl_feats_lst[["CSM.X"]] <- c(NULL
#     , <comma-separated-features-vector>
#                                   )
# dAFeats.CSM.X %<d-% c(NULL
#     # Interaction feats up to varImp(RFE.X.glmnet) >= 50
#     , <comma-separated-features-vector>
#     , setdiff(myextract_actual_feats(predictors(rfe_fit_results)), c(NULL
#                , <comma-separated-features-vector>
#                                                                       ))    
#                                   )
# glb_mdl_feats_lst[["CSM.X"]] <- "%<d-% dAFeats.CSM.X"

glbMdlFamilies[["Final"]] <- c(NULL) # NULL vector acceptable

glbMdlAllowParallel <- list()
#glbMdlAllowParallel[["<mdlId>"]] <- FALSE
glbMdlAllowParallel[["Max.cor.Y##rcv#rpart"]] <- FALSE
glbMdlAllowParallel[["All.X##rcv#glm"]] <- FALSE

# Check if tuning parameters make fit better; make it mdlFamily customizable ?
glbMdlTuneParams <- data.frame()
# When glmnet crashes at model$grid with error: ???
glmnetTuneParams <- rbind(data.frame()
                        ,data.frame(parameter = "alpha",  vals = "0.100 0.325 0.550 0.775 1.000")
                        ,data.frame(parameter = "lambda", vals = "9.342e-02")    
                        )
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams,
#                                cbind(data.frame(mdlId = "<mdlId>"),
#                                      glmnetTuneParams))

    #avNNet    
    #   size=[1] 3 5 7 9; decay=[0] 1e-04 0.001  0.01   0.1; bag=[FALSE]; RMSE=1.3300906 

    #bagEarth
    #   degree=1 [2] 3; nprune=64 128 256 512 [1024]; RMSE=0.6486663 (up)
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "bagEarth", parameter = "nprune", vals = "256")
#     ,data.frame(method = "bagEarth", parameter = "degree", vals = "2")    
# ))

    #earth 
    #   degree=[1]; nprune=2  [9] 17 25 33; RMSE=0.1334478
    
    #gbm 
    #   shrinkage=0.05 [0.10] 0.15 0.20 0.25; n.trees=100 150 200 [250] 300; interaction.depth=[1] 2 3 4 5; n.minobsinnode=[10]; RMSE=0.2008313     
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "gbm", parameter = "shrinkage", min = 0.05, max = 0.25, by = 0.05)
#     ,data.frame(method = "gbm", parameter = "n.trees", min = 100, max = 300, by = 50)
#     ,data.frame(method = "gbm", parameter = "interaction.depth", min = 1, max = 5, by = 1)
#     ,data.frame(method = "gbm", parameter = "n.minobsinnode", min = 10, max = 10, by = 10)
#     #seq(from=0.05,  to=0.25, by=0.05)
# ))

    #glmnet
    #   alpha=0.100 [0.325] 0.550 0.775 1.000; lambda=0.0005232693 0.0024288010 0.0112734954 [0.0523269304] 0.2428800957; RMSE=0.6164891
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "glmnet", parameter = "alpha", vals = "0.550 0.775 0.8875 0.94375 1.000")
#     ,data.frame(method = "glmnet", parameter = "lambda", vals = "9.858855e-05 0.0001971771 0.0009152152 0.0042480525 0.0197177130")    
# ))

    #nnet    
    #   size=3 5 [7] 9 11; decay=0.0001 0.001 0.01 [0.1] 0.2; RMSE=0.9287422
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "nnet", parameter = "size", vals = "3 5 7 9 11")
#     ,data.frame(method = "nnet", parameter = "decay", vals = "0.0001 0.0010 0.0100 0.1000 0.2000")    
# ))

    #rf # Don't bother; results are not deterministic
    #       mtry=2  35  68 [101] 134; RMSE=0.1339974
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "rf", parameter = "mtry", vals = "2 5 9 13 17")
# ))

    #rpart 
    #   cp=0.020 [0.025] 0.030 0.035 0.040; RMSE=0.1770237
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()    
#     ,data.frame(method = "rpart", parameter = "cp", vals = "0.004347826 0.008695652 0.017391304 0.021739130 0.034782609")
# ))
    
    #svmLinear
    #   C=0.01 0.05 [0.10] 0.50 1.00 2.00 3.00 4.00; RMSE=0.1271318; 0.1296718
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "svmLinear", parameter = "C", vals = "0.01 0.05 0.1 0.5 1")
# ))

    #svmLinear2    
    #   cost=0.0625 0.1250 [0.25] 0.50 1.00; RMSE=0.1276354 
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method = "svmLinear2", parameter = "cost", vals = "0.0625 0.125 0.25 0.5 1")
# ))

    #svmPoly    
    #   degree=[1] 2 3 4 5; scale=0.01 0.05 [0.1] 0.5 1; C=0.50 1.00 [2.00] 3.00 4.00; RMSE=0.1276130
# glbMdlTuneParams <- myrbind_df(glbMdlTuneParams, rbind(data.frame()
#     ,data.frame(method="svmPoly", parameter="degree", min=1, max=5, by=1) #seq(1, 5, 1)
#     ,data.frame(method="svmPoly", parameter="scale", vals="0.01, 0.05, 0.1, 0.5, 1")
#     ,data.frame(method="svmPoly", parameter="C", vals="0.50, 1.00, 2.00, 3.00, 4.00")    
# ))

    #svmRadial
    #   sigma=[0.08674323]; C=0.25 0.50 1.00 [2.00] 4.00; RMSE=0.1614957
    
#glb2Sav(); all.equal(sav_models_df, glb_models_df)
    
glb_preproc_methods <- NULL
#     c("YeoJohnson", "center.scale", "range", "pca", "ica", "spatialSign")

# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<feat>")

glbMdlMetric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glbMdlMetricSummary <- NULL # or "<metric_name>"
glbMdlMetricMaximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glbMdlMetricSummaryFn <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glbMdlMetric_terms)
#     metric <- sum(confusion_mtrx * glbMdlMetric_terms) / nrow(data)
#     names(metric) <- glbMdlMetricSummary
#     return(metric)
# }

glbMdlCheckRcv <- FALSE # Turn it on when needed; otherwise takes long time
glb_rcv_n_folds <- 3 # or NULL
glb_rcv_n_repeats <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glbMdlMetricsEval <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit", "min.RMSE.fit")
    #glbMdlMetricsEval <- c("min.RMSE.fit", "max.R.sq.fit", "max.Adj.R.sq.fit")    
if (glb_is_classification) {
    if (glb_is_binomial)
        glbMdlMetricsEval <- 
            c("max.Accuracy.OOB", "max.AUCROCR.OOB", "max.AUCpROC.OOB", "min.aic.fit", "max.Accuracy.fit") else        
        glbMdlMetricsEval <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

# select from NULL [no ensemble models], "auto" [all models better than MFO or Baseline], c(mdl_ids in glb_models_lst) [Typically top-rated models in auto]
glb_mdl_ensemble <- NULL
#     "%<d-% setdiff(mygetEnsembleAutoMdlIds(), 'CSM.X.rf')" 
#     c(<comma-separated-mdlIds>
#      )

# Only for classifications; for regressions remove "(.*)\\.prob" form the regex
# tmp_fitobs_df <- glbObsFit[, grep(paste0("^", gsub(".", "\\.", mygetPredictIds$value, fixed = TRUE), "CSM\\.X\\.(.*)\\.prob"), names(glbObsFit), value = TRUE)]; cor_mtrx <- cor(tmp_fitobs_df); cor_vctr <- sort(cor_mtrx[row.names(orderBy(~-Overall, varImp(glb_models_lst[["Ensemble.repeatedcv.glmnet"]])$imp))[1], ]); summary(cor_vctr); cor_vctr
#ntv.glm <- glm(reformulate(indep_vars, glb_rsp_var), family = "binomial", data = glbObsFit)
#step.glm <- step(ntv.glm)

glb_sel_mdl_id <- "All.X##rcv#glmnet" #select from c(NULL, "All.X##rcv#glmnet", "RFE.X##rcv#glmnet", <mdlId>)
glb_fin_mdl_id <- NULL #select from c(NULL, glb_sel_mdl_id)

glb_dsp_cols <- c(glbFeatsId, glbFeatsCategory, glb_rsp_var
#               List critical cols excl. glbFeatsId, glbFeatsCategory & glb_rsp_var
                  )

# Output specs
glbOutDataVizFname <- NULL # choose from c(NULL, "<projectId>_obsall.csv")
glb_out_obs <- NULL # select from c(NULL : default to "new", "all", "new", "trn")
glb_out_vars_lst <- list()
# glbFeatsId will be the first output column, by default

if (glb_is_classification && glb_is_binomial) {
    glb_out_vars_lst[["Probability1"]] <- 
        "%<d-% mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$prob" 
} else {
    glb_out_vars_lst[[glb_rsp_var]] <- 
        "%<d-% mygetPredictIds(glb_rsp_var, glb_fin_mdl_id)$value"
}    
# glb_out_vars_lst[[glb_rsp_var_raw]] <- glb_rsp_var_raw
# glb_out_vars_lst[[paste0(head(unlist(strsplit(mygetPredictIds$value, "")), -1), collapse = "")]] <-

glbOutStackFnames <- NULL #: default
    # c("ebayipads_txt_assoc1_out_bid1_stack.csv") # manual stack
    # c("ebayipads_finmdl_bid1_out_nnet_1.csv") # universal stack
glb_out_pfx <- "Houses_tmplt_"
glb_save_envir <- FALSE # or TRUE

# Depict process
glb_analytics_pn <- petrinet(name = "glb_analytics_pn",
                        trans_df = data.frame(id = 1:6,
    name = c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df = data.frame(
    begin = c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end   = c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](Houses_tmplt_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_chunks_df <- myadd_chunk(NULL, "import.data")
```

```
##         label step_major step_minor label_minor   bgn end elapsed
## 1 import.data          1          0           0 9.069  NA      NA
```

## Step `1.0: import data`
#### chunk option: eval=<r condition>

```
## [1] "Reading file ./data/home_data.csv..."
## [1] "dimensions of data in ./data/home_data.csv: 21,613 rows x 21 cols"
##           id            date   price bedrooms bathrooms sqft_living
## 1 7129300520 20141013T000000  221900        3      1.00        1180
## 2 6414100192 20141209T000000  538000        3      2.25        2570
## 3 5631500400 20150225T000000  180000        2      1.00         770
## 4 2487200875 20141209T000000  604000        4      3.00        1960
## 5 1954400510 20150218T000000  510000        3      2.00        1680
## 6 7237550310 20140512T000000 1225000        4      4.50        5420
##   sqft_lot floors waterfront view condition grade sqft_above sqft_basement
## 1     5650      1          0    0         3     7       1180             0
## 2     7242      2          0    0         3     7       2170           400
## 3    10000      1          0    0         3     6        770             0
## 4     5000      1          0    0         5     7       1050           910
## 5     8080      1          0    0         3     8       1680             0
## 6   101930      1          0    0         3    11       3890          1530
##   yr_built yr_renovated zipcode     lat     long sqft_living15 sqft_lot15
## 1     1955            0   98178 47.5112 -122.257          1340       5650
## 2     1951         1991   98125 47.7210 -122.319          1690       7639
## 3     1933            0   98028 47.7379 -122.233          2720       8062
## 4     1965            0   98136 47.5208 -122.393          1360       5000
## 5     1987            0   98074 47.6168 -122.045          1800       7503
## 6     2001            0   98053 47.6561 -122.005          4760     101930
##               id            date  price bedrooms bathrooms sqft_living
## 747   9277200111 20140714T000000 650000        4      1.75        2010
## 3293  7701961030 20150129T000000 875000        4      2.50        3600
## 11006 8651480550 20140623T000000 600000        3      2.50        2260
## 15728 6116500290 20140714T000000 799950        6      2.75        3040
## 15897 2131701410 20150427T000000 299950        3      2.25        1370
## 21390 3448740360 20150429T000000 418500        4      2.50        2190
##       sqft_lot floors waterfront view condition grade sqft_above
## 747       5070      1          0    1         4     7       1300
## 3293     21794      2          0    0         4    11       3600
## 11006    10153      2          0    0         3    10       2260
## 15728    36721      1          0    3         4     9       1760
## 15897     5000      2          0    0         3     7       1370
## 21390     4866      2          0    0         3     7       2190
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 747             710     1963            0   98116 47.5793 -122.402
## 3293              0     1990            0   98077 47.7121 -122.072
## 11006             0     1987            0   98074 47.6410 -122.068
## 15728          1280     1958            0   98166 47.4488 -122.356
## 15897             0     1990            0   98019 47.7372 -121.981
## 21390             0     2009            0   98059 47.4907 -122.152
##       sqft_living15 sqft_lot15
## 747            2180       5400
## 3293           3410      19864
## 11006          2740      10153
## 15728          2420      21075
## 15897          1600       7724
## 21390          2190       5670
##               id            date  price bedrooms bathrooms sqft_living
## 21608 2997800021 20150219T000000 475000        3      2.50        1310
## 21609  263000018 20140521T000000 360000        3      2.50        1530
## 21610 6600060120 20150223T000000 400000        4      2.50        2310
## 21611 1523300141 20140623T000000 402101        2      0.75        1020
## 21612  291310100 20150116T000000 400000        3      2.50        1600
## 21613 1523300157 20141015T000000 325000        2      0.75        1020
##       sqft_lot floors waterfront view condition grade sqft_above
## 21608     1294      2          0    0         3     8       1180
## 21609     1131      3          0    0         3     8       1530
## 21610     5813      2          0    0         3     8       2310
## 21611     1350      2          0    0         3     7       1020
## 21612     2388      2          0    0         3     8       1600
## 21613     1076      2          0    0         3     7       1020
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 21608           130     2008            0   98116 47.5773 -122.409
## 21609             0     2009            0   98103 47.6993 -122.346
## 21610             0     2014            0   98146 47.5107 -122.362
## 21611             0     2009            0   98144 47.5944 -122.299
## 21612             0     2004            0   98027 47.5345 -122.069
## 21613             0     2008            0   98144 47.5941 -122.299
##       sqft_living15 sqft_lot15
## 21608          1330       1265
## 21609          1530       1509
## 21610          1830       7200
## 21611          1020       2007
## 21612          1410       1287
## 21613          1020       1357
## 'data.frame':	21613 obs. of  20 variables:
##  $ id           : num  7.13e+09 6.41e+09 5.63e+09 2.49e+09 1.95e+09 ...
##  $ date         : chr  "20141013T000000" "20141209T000000" "20150225T000000" "20141209T000000" ...
##  $ price        : int  221900 538000 180000 604000 510000 1225000 257500 291850 229500 323000 ...
##  $ bedrooms     : int  3 3 2 4 3 4 3 3 3 3 ...
##  $ bathrooms    : num  1 2.25 1 3 2 4.5 2.25 1.5 1 2.5 ...
##  $ sqft_living  : int  1180 2570 770 1960 1680 5420 1715 1060 1780 1890 ...
##  $ sqft_lot     : int  5650 7242 10000 5000 8080 101930 6819 9711 7470 6560 ...
##  $ floors       : num  1 2 1 1 1 1 2 1 1 2 ...
##  $ waterfront   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ view         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ condition    : int  3 3 3 5 3 3 3 3 3 3 ...
##  $ grade        : int  7 7 6 7 8 11 7 7 7 7 ...
##  $ sqft_above   : int  1180 2170 770 1050 1680 3890 1715 1060 1050 1890 ...
##  $ sqft_basement: int  0 400 0 910 0 1530 0 0 730 0 ...
##  $ yr_built     : int  1955 1951 1933 1965 1987 2001 1995 1963 1960 2003 ...
##  $ yr_renovated : int  0 1991 0 0 0 0 0 0 0 0 ...
##  $ zipcode      : int  98178 98125 98028 98136 98074 98053 98003 98198 98146 98038 ...
##  $ lat          : num  47.5 47.7 47.7 47.5 47.6 ...
##  $ long         : num  -122 -122 -122 -122 -122 ...
##  $ sqft_living15: int  1340 1690 2720 1360 1800 4760 2238 1650 1780 2390 ...
## NULL
## 'data.frame':	21613 obs. of  21 variables:
##  $ id           : num  7.13e+09 6.41e+09 5.63e+09 2.49e+09 1.95e+09 ...
##  $ date         : chr  "20141013T000000" "20141209T000000" "20150225T000000" "20141209T000000" ...
##  $ price        : int  221900 538000 180000 604000 510000 1225000 257500 291850 229500 323000 ...
##  $ bedrooms     : int  3 3 2 4 3 4 3 3 3 3 ...
##  $ bathrooms    : num  1 2.25 1 3 2 4.5 2.25 1.5 1 2.5 ...
##  $ sqft_living  : int  1180 2570 770 1960 1680 5420 1715 1060 1780 1890 ...
##  $ sqft_lot     : int  5650 7242 10000 5000 8080 101930 6819 9711 7470 6560 ...
##  $ floors       : num  1 2 1 1 1 1 2 1 1 2 ...
##  $ waterfront   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ view         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ condition    : int  3 3 3 5 3 3 3 3 3 3 ...
##  $ grade        : int  7 7 6 7 8 11 7 7 7 7 ...
##  $ sqft_above   : int  1180 2170 770 1050 1680 3890 1715 1060 1050 1890 ...
##  $ sqft_basement: int  0 400 0 910 0 1530 0 0 730 0 ...
##  $ yr_built     : int  1955 1951 1933 1965 1987 2001 1995 1963 1960 2003 ...
##  $ yr_renovated : int  0 1991 0 0 0 0 0 0 0 0 ...
##  $ zipcode      : int  98178 98125 98028 98136 98074 98053 98003 98198 98146 98038 ...
##  $ lat          : num  47.5 47.7 47.7 47.5 47.6 ...
##  $ long         : num  -122 -122 -122 -122 -122 ...
##  $ sqft_living15: int  1340 1690 2720 1360 1800 4760 2238 1650 1780 2390 ...
##  $ sqft_lot15   : int  5650 7639 8062 5000 7503 101930 6819 9711 8113 7570 ...
## NULL
```

```
## Warning in myprint_str_df(df): [list output truncated]
```

```
## Loading required package: caTools
```

```
##            id            date  price bedrooms bathrooms sqft_living
## 4  2487200875 20141209T000000 604000        4       3.0        1960
## 17 1875500060 20140731T000000 395000        3       2.0        1890
## 23 7137970340 20140703T000000 285000        5       2.5        2270
## 28 3303700376 20141201T000000 667000        3       1.0        1400
## 30 1873100390 20150302T000000 719000        4       2.5        2570
## 32 2426039314 20141201T000000 280000        2       1.5        1190
##    sqft_lot floors waterfront view condition grade sqft_above
## 4      5000    1.0          0    0         5     7       1050
## 17    14040    2.0          0    0         3     7       1890
## 23     6300    2.0          0    0         3     8       2270
## 28     1581    1.5          0    0         5     8       1400
## 30     7173    2.0          0    0         3     8       2570
## 32     1265    3.0          0    0         3     7       1190
##    sqft_basement yr_built yr_renovated zipcode     lat     long
## 4            910     1965            0   98136 47.5208 -122.393
## 17             0     1994            0   98019 47.7277 -121.962
## 23             0     1995            0   98092 47.3266 -122.169
## 28             0     1909            0   98112 47.6221 -122.314
## 30             0     2005            0   98052 47.7073 -122.110
## 32             0     2005            0   98133 47.7274 -122.357
##    sqft_living15 sqft_lot15
## 4           1360       5000
## 17          1890      14018
## 23          2240       7005
## 28          1860       3861
## 30          2630       6026
## 32          1390       1756
##               id            date  price bedrooms bathrooms sqft_living
## 6707  9109000050 20140709T000000 275000        3      1.00        1200
## 8429   263000329 20141008T000000 349950        3      2.50        1420
## 12423 2172000570 20140624T000000 317000        5      2.50        2360
## 12802 7701700040 20140925T000000 320000        3      1.75        1510
## 13577 9269200150 20140715T000000 390000        1      1.75        1440
## 17032 2742100009 20140506T000000 385000        3      1.75        1900
##       sqft_lot floors waterfront view condition grade sqft_above
## 6707      7800    1.0          0    0         4     7       1200
## 8429      1162    3.0          0    0         3     8       1420
## 12423    11375    1.0          0    0         4     7       1180
## 12802    30185    1.5          0    0         3     7       1510
## 13577     4920    1.0          0    0         3     7        720
## 17032     5520    1.0          0    0         3     7       1280
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 6707              0     1954            0   98126 47.5196 -122.371
## 8429              0     2002            0   98103 47.6982 -122.349
## 12423          1180     1962            0   98178 47.4875 -122.255
## 12802             0     1976            0   98058 47.4118 -122.089
## 13577           720     1923            0   98126 47.5340 -122.376
## 17032           620     1982            0   98118 47.5549 -122.292
##       sqft_living15 sqft_lot15
## 6707           1230       7070
## 8429           1430       1560
## 12423          1160       7800
## 12802          1470      12465
## 13577          1440       4920
## 17032          1330       5196
##               id            date  price bedrooms bathrooms sqft_living
## 21593 1931300412 20150416T000000 475000        3      2.25        1190
## 21597 7502800100 20140813T000000 679950        5      2.75        3600
## 21602 5100403806 20150407T000000 467000        3      2.50        1425
## 21603  844000965 20140626T000000 224000        3      1.75        1500
## 21610 6600060120 20150223T000000 400000        4      2.50        2310
## 21613 1523300157 20141015T000000 325000        2      0.75        1020
##       sqft_lot floors waterfront view condition grade sqft_above
## 21593     1200      3          0    0         3     8       1190
## 21597     9437      2          0    0         3     9       3600
## 21602     1179      3          0    0         3     8       1425
## 21603    11968      1          0    0         3     6       1500
## 21610     5813      2          0    0         3     8       2310
## 21613     1076      2          0    0         3     7       1020
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 21593             0     2008            0   98103 47.6542 -122.346
## 21597             0     2014            0   98059 47.4822 -122.131
## 21602             0     2008            0   98125 47.6963 -122.318
## 21603             0     2014            0   98010 47.3095 -122.002
## 21610             0     2014            0   98146 47.5107 -122.362
## 21613             0     2008            0   98144 47.5941 -122.299
##       sqft_living15 sqft_lot15
## 21593          1180       1224
## 21597          3550       9421
## 21602          1285       1253
## 21603          1320      11303
## 21610          1830       7200
## 21613          1020       1357
## 'data.frame':	3772 obs. of  21 variables:
##  $ id           : num  2.49e+09 1.88e+09 7.14e+09 3.30e+09 1.87e+09 ...
##  $ date         : chr  "20141209T000000" "20140731T000000" "20140703T000000" "20141201T000000" ...
##  $ price        : int  604000 395000 285000 667000 719000 280000 640000 785000 345000 600000 ...
##  $ bedrooms     : int  4 3 5 3 4 2 4 4 5 3 ...
##  $ bathrooms    : num  3 2 2.5 1 2.5 1.5 2 2.5 2.5 1.75 ...
##  $ sqft_living  : int  1960 1890 2270 1400 2570 1190 2360 2290 3150 1410 ...
##  $ sqft_lot     : int  5000 14040 6300 1581 7173 1265 6000 13416 9134 4080 ...
##  $ floors       : num  1 2 2 1.5 2 3 2 2 1 1 ...
##  $ waterfront   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ view         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ condition    : int  5 3 3 5 3 3 4 4 4 4 ...
##  $ grade        : int  7 7 8 8 8 7 8 9 8 7 ...
##  $ sqft_above   : int  1050 1890 2270 1400 2570 1190 2360 2290 1640 1000 ...
##  $ sqft_basement: int  910 0 0 0 0 0 0 0 1510 410 ...
##  $ yr_built     : int  1965 1994 1995 1909 2005 2005 1904 1981 1966 1950 ...
##  $ yr_renovated : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ zipcode      : int  98136 98019 98092 98112 98052 98133 98107 98007 98056 98117 ...
##  $ lat          : num  47.5 47.7 47.3 47.6 47.7 ...
##  $ long         : num  -122 -122 -122 -122 -122 ...
##  $ sqft_living15: int  1360 1890 2240 1860 2630 1390 1730 2680 1990 1410 ...
##  $ sqft_lot15   : int  5000 14018 7005 3861 6026 1756 4700 13685 9133 4080 ...
##  - attr(*, "comment")= chr "glbObsNew"
##           id            date   price bedrooms bathrooms sqft_living
## 1 7129300520 20141013T000000  221900        3      1.00        1180
## 2 6414100192 20141209T000000  538000        3      2.25        2570
## 3 5631500400 20150225T000000  180000        2      1.00         770
## 5 1954400510 20150218T000000  510000        3      2.00        1680
## 6 7237550310 20140512T000000 1225000        4      4.50        5420
## 7 1321400060 20140627T000000  257500        3      2.25        1715
##   sqft_lot floors waterfront view condition grade sqft_above sqft_basement
## 1     5650      1          0    0         3     7       1180             0
## 2     7242      2          0    0         3     7       2170           400
## 3    10000      1          0    0         3     6        770             0
## 5     8080      1          0    0         3     8       1680             0
## 6   101930      1          0    0         3    11       3890          1530
## 7     6819      2          0    0         3     7       1715             0
##   yr_built yr_renovated zipcode     lat     long sqft_living15 sqft_lot15
## 1     1955            0   98178 47.5112 -122.257          1340       5650
## 2     1951         1991   98125 47.7210 -122.319          1690       7639
## 3     1933            0   98028 47.7379 -122.233          2720       8062
## 5     1987            0   98074 47.6168 -122.045          1800       7503
## 6     2001            0   98053 47.6561 -122.005          4760     101930
## 7     1995            0   98003 47.3097 -122.327          2238       6819
##               id            date  price bedrooms bathrooms sqft_living
## 813   4307350200 20150512T000000 347000        3       2.5        1680
## 1260  2798600240 20141114T000000 295700        4       2.5        1720
## 4438  3874900090 20150326T000000 448000        2       2.0        1670
## 10258 1995200200 20140506T000000 313950        3       1.0        1510
## 12776 7812800155 20150318T000000 170000        3       1.0         790
## 17617 4036800900 20140924T000000 447000        3       1.0        1310
##       sqft_lot floors waterfront view condition grade sqft_above
## 813       4308      2          0    0         3     7       1680
## 1260      5805      2          0    0         3     8       1720
## 4438      7772      1          0    0         4     6        860
## 10258     6083      1          0    0         4     6        860
## 12776     6750      1          0    0         2     6        790
## 17617     7000      1          0    0         4     7       1310
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 813               0     2004            0   98056 47.4802 -122.179
## 1260              0     2000            0   98092 47.3286 -122.208
## 4438            810     1919            0   98126 47.5461 -122.377
## 10258           650     1940            0   98115 47.6966 -122.324
## 12776             0     1944            0   98178 47.4984 -122.240
## 17617             0     1958            0   98008 47.6019 -122.123
##       sqft_living15 sqft_lot15
## 813            2160       4182
## 1260           2360       7700
## 4438           1300       7770
## 10258          1510       5712
## 12776           960       6298
## 17617          1280       7300
##               id            date   price bedrooms bathrooms sqft_living
## 21606 3448900210 20141014T000000  610685        4      2.50        2520
## 21607 7936000429 20150326T000000 1007500        4      3.50        3510
## 21608 2997800021 20150219T000000  475000        3      2.50        1310
## 21609  263000018 20140521T000000  360000        3      2.50        1530
## 21611 1523300141 20140623T000000  402101        2      0.75        1020
## 21612  291310100 20150116T000000  400000        3      2.50        1600
##       sqft_lot floors waterfront view condition grade sqft_above
## 21606     6023      2          0    0         3     9       2520
## 21607     7200      2          0    0         3     9       2600
## 21608     1294      2          0    0         3     8       1180
## 21609     1131      3          0    0         3     8       1530
## 21611     1350      2          0    0         3     7       1020
## 21612     2388      2          0    0         3     8       1600
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 21606             0     2014            0   98056 47.5137 -122.167
## 21607           910     2009            0   98136 47.5537 -122.398
## 21608           130     2008            0   98116 47.5773 -122.409
## 21609             0     2009            0   98103 47.6993 -122.346
## 21611             0     2009            0   98144 47.5944 -122.299
## 21612             0     2004            0   98027 47.5345 -122.069
##       sqft_living15 sqft_lot15
## 21606          2520       6023
## 21607          2050       6200
## 21608          1330       1265
## 21609          1530       1509
## 21611          1020       2007
## 21612          1410       1287
## 'data.frame':	17841 obs. of  21 variables:
##  $ id           : num  7.13e+09 6.41e+09 5.63e+09 1.95e+09 7.24e+09 ...
##  $ date         : chr  "20141013T000000" "20141209T000000" "20150225T000000" "20150218T000000" ...
##  $ price        : int  221900 538000 180000 510000 1225000 257500 291850 229500 323000 662500 ...
##  $ bedrooms     : int  3 3 2 3 4 3 3 3 3 3 ...
##  $ bathrooms    : num  1 2.25 1 2 4.5 2.25 1.5 1 2.5 2.5 ...
##  $ sqft_living  : int  1180 2570 770 1680 5420 1715 1060 1780 1890 3560 ...
##  $ sqft_lot     : int  5650 7242 10000 8080 101930 6819 9711 7470 6560 9796 ...
##  $ floors       : num  1 2 1 1 1 2 1 1 2 1 ...
##  $ waterfront   : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ view         : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ condition    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ grade        : int  7 7 6 8 11 7 7 7 7 8 ...
##  $ sqft_above   : int  1180 2170 770 1680 3890 1715 1060 1050 1890 1860 ...
##  $ sqft_basement: int  0 400 0 0 1530 0 0 730 0 1700 ...
##  $ yr_built     : int  1955 1951 1933 1987 2001 1995 1963 1960 2003 1965 ...
##  $ yr_renovated : int  0 1991 0 0 0 0 0 0 0 0 ...
##  $ zipcode      : int  98178 98125 98028 98074 98053 98003 98198 98146 98038 98007 ...
##  $ lat          : num  47.5 47.7 47.7 47.6 47.7 ...
##  $ long         : num  -122 -122 -122 -122 -122 ...
##  $ sqft_living15: int  1340 1690 2720 1800 4760 2238 1650 1780 2390 2210 ...
##  $ sqft_lot15   : int  5650 7639 8062 7503 101930 6819 9711 8113 7570 8925 ...
##  - attr(*, "comment")= chr "glbObsTrn"
```

```
## [1] "Creating new feature: .pos..."
## [1] "Creating new feature: id.date..."
```

```
## [1] "Partition stats:"
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##        price.cut.fctr  .src    .n
## 1 (6.74e+04,2.62e+06] Train 17767
## 2 (6.74e+04,2.62e+06]  Test  3764
## 3 (2.62e+06,5.16e+06] Train    68
## 4 (2.62e+06,5.16e+06]  Test     8
## 5 (5.16e+06,7.71e+06] Train     6
##        price.cut.fctr  .src    .n
## 1 (6.74e+04,2.62e+06] Train 17767
## 2 (6.74e+04,2.62e+06]  Test  3764
## 3 (2.62e+06,5.16e+06] Train    68
## 4 (2.62e+06,5.16e+06]  Test     8
## 5 (5.16e+06,7.71e+06] Train     6
```

![](Houses_tmplt_files/figure-html/import.data-1.png) 

```
##    .src    .n
## 1 Train 17841
## 2  Test  3772
```

```
## Loading required package: lazyeval
## Loading required package: gdata
## gdata: read.xls support for 'XLS' (Excel 97-2004) files ENABLED.
## 
## gdata: read.xls support for 'XLSX' (Excel 2007+) files ENABLED.
## 
## Attaching package: 'gdata'
## 
## The following objects are masked from 'package:dplyr':
## 
##     combine, first, last
## 
## The following object is masked from 'package:stats':
## 
##     nobs
## 
## The following object is masked from 'package:utils':
## 
##     object.size
```

```
## [1] "Found 0 duplicates by all features:"
```

```
## NULL
```

```
##          label step_major step_minor label_minor    bgn    end elapsed
## 1  import.data          1          0           0  9.069 18.917   9.848
## 2 inspect.data          2          0           0 18.917     NA      NA
```

## Step `2.0: inspect data`

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](Houses_tmplt_files/figure-html/inspect.data-1.png) 

```
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##      bedrooms     bathrooms    waterfront          view sqft_basement 
##            13            10         21450         19489         13126 
##  yr_renovated 
##         20699 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
##    date id.date 
##       0       0
```

![](Houses_tmplt_files/figure-html/inspect.data-2.png) ![](Houses_tmplt_files/figure-html/inspect.data-3.png) 

```
##          label step_major step_minor label_minor    bgn    end elapsed
## 2 inspect.data          2          0           0 18.917 26.933   8.017
## 3   scrub.data          2          1           1 26.934     NA      NA
```

### Step `2.1: scrub data`

```
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##      bedrooms     bathrooms    waterfront          view sqft_basement 
##            13            10         21450         19489         13126 
##  yr_renovated 
##         20699 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
##    date id.date 
##       0       0
```

```
##            label step_major step_minor label_minor    bgn    end elapsed
## 3     scrub.data          2          1           1 26.934 29.275   2.341
## 4 transform.data          2          2           2 29.276     NA      NA
```

### Step `2.2: transform data`

```
##              label step_major step_minor label_minor    bgn    end elapsed
## 4   transform.data          2          2           2 29.276 29.313   0.038
## 5 extract.features          3          0           0 29.314     NA      NA
```

## Step `3.0: extract features`

```
##                     label step_major step_minor label_minor    bgn    end
## 5        extract.features          3          0           0 29.314 29.333
## 6 extract.features.string          3          1           1 29.333     NA
##   elapsed
## 5   0.019
## 6      NA
```

### Step `3.1: extract features string`

```
##                         label step_major step_minor label_minor   bgn end
## 1 extract.features.string.bgn          1          0           0 29.36  NA
##   elapsed
## 1      NA
```

```
##                                       label step_major step_minor
## 1               extract.features.string.bgn          1          0
## 2 extract.features.stringfactorize.str.vars          2          0
##   label_minor   bgn   end elapsed
## 1           0 29.36 29.37    0.01
## 2           0 29.37    NA      NA
```

```
##      date      .src   id.date 
##    "date"    ".src" "id.date"
```

```
##                       label step_major step_minor label_minor    bgn
## 6   extract.features.string          3          1           1 29.333
## 7 extract.features.datetime          3          2           2 29.384
##      end elapsed
## 6 29.383   0.051
## 7     NA      NA
```

### Step `3.2: extract features datetime`

```
##                           label step_major step_minor label_minor    bgn
## 1 extract.features.datetime.bgn          1          0           0 29.411
##   end elapsed
## 1  NA      NA
```

```
##                       label step_major step_minor label_minor    bgn   end
## 7 extract.features.datetime          3          2           2 29.384 29.42
## 8    extract.features.price          3          3           3 29.420    NA
##   elapsed
## 7   0.036
## 8      NA
```

### Step `3.3: extract features price`

```
##                        label step_major step_minor label_minor    bgn end
## 1 extract.features.price.bgn          1          0           0 29.445  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 8 extract.features.price          3          3           3 29.420 29.454
## 9  extract.features.text          3          4           4 29.454     NA
##   elapsed
## 8   0.034
## 9      NA
```

### Step `3.4: extract features text`

```
##                       label step_major step_minor label_minor    bgn end
## 1 extract.features.text.bgn          1          0           0 29.496  NA
##   elapsed
## 1      NA
```

```
##                    label step_major step_minor label_minor    bgn    end
## 9  extract.features.text          3          4           4 29.454 29.506
## 10  extract.features.end          3          5           5 29.506     NA
##    elapsed
## 9    0.052
## 10      NA
```

### Step `3.5: extract features end`

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Houses_tmplt_files/figure-html/extract.features.end-1.png) 

```
##                   label step_major step_minor label_minor    bgn   end
## 10 extract.features.end          3          5           5 29.506 30.39
## 11  manage.missing.data          4          0           0 30.391    NA
##    elapsed
## 10   0.884
## 11      NA
```

### Step `4.0: manage missing data`

```
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##      bedrooms     bathrooms    waterfront          view sqft_basement 
##            13            10         21450         19489         13126 
##  yr_renovated 
##         20699 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
##    date id.date 
##       0       0
```

```
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##      bedrooms     bathrooms    waterfront          view sqft_basement 
##            13            10         21450         19489         13126 
##  yr_renovated 
##         20699 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
##    date id.date 
##       0       0
```

```
##                  label step_major step_minor label_minor    bgn    end
## 11 manage.missing.data          4          0           0 30.391 30.705
## 12        cluster.data          5          0           0 30.706     NA
##    elapsed
## 11   0.314
## 12      NA
```

## Step `5.0: cluster data`

```
##                      label step_major step_minor label_minor    bgn    end
## 12            cluster.data          5          0           0 30.706 30.774
## 13 partition.data.training          6          0           0 30.774     NA
##    elapsed
## 12   0.068
## 13      NA
```

## Step `6.0: partition data training`

```
## [1] "Newdata contains non-NA data for price; setting OOB to Newdata"
```

```
##   .category .n.Fit .n.OOB .n.Tst .freqRatio.Fit .freqRatio.OOB
## 1    .dummy  17841   3772   3772              1              1
##   .freqRatio.Tst
## 1              1
```

```
## [1] "glbObsAll: "
```

```
## [1] 21613    27
```

```
## [1] "glbObsTrn: "
```

```
## [1] 17841    27
```

```
## [1] "glbObsFit: "
```

```
## [1] 17841    26
```

```
## [1] "glbObsOOB: "
```

```
## [1] 3772   26
```

```
## [1] "glbObsNew: "
```

```
## [1] 3772   26
```

```
##                      label step_major step_minor label_minor    bgn   end
## 13 partition.data.training          6          0           0 30.774 31.31
## 14         select.features          7          0           0 31.311    NA
##    elapsed
## 13   0.537
## 14      NA
```

## Step `7.0: select features`

```
## Warning in cor(data.matrix(entity_df[, sel_feats]), y =
## as.numeric(entity_df[, : the standard deviation is zero
```

```
## Loading required package: reshape2
```

```
##                      cor.y exclude.as.feat   cor.y.abs cor.high.X
## sqft_living    0.708441487               1 0.708441487         NA
## grade          0.668890914               1 0.668890914         NA
## sqft_above     0.610988429               1 0.610988429         NA
## sqft_living15  0.586792062               1 0.586792062         NA
## bathrooms      0.530073438               1 0.530073438         NA
## view           0.406939514               1 0.406939514         NA
## sqft_basement  0.331786255               1 0.331786255         NA
## bedrooms       0.310672803               1 0.310672803         NA
## lat            0.302852840               1 0.302852840         NA
## waterfront     0.278224722               1 0.278224722         NA
## floors         0.256425528               1 0.256425528         NA
## yr_renovated   0.122721006               1 0.122721006         NA
## sqft_lot       0.090440967               1 0.090440967         NA
## sqft_lot15     0.084523271               1 0.084523271         NA
## yr_built       0.056360729               1 0.056360729         NA
## condition      0.036979576               1 0.036979576         NA
## .pos           0.028282285               0 0.028282285         NA
## long           0.022417163               1 0.022417163         NA
## .rnorm        -0.007611976               0 0.007611976         NA
## id            -0.021573175               1 0.021573175         NA
## zipcode       -0.053505492               1 0.053505492         NA
## .category               NA               1          NA         NA
##                freqRatio percentUnique zeroVar   nzv is.cor.y.abs.low
## sqft_living     1.035398  5.403285e+00   FALSE FALSE            FALSE
## grade           1.463147  6.726080e-02   FALSE FALSE            FALSE
## sqft_above      1.017143  4.898829e+00   FALSE FALSE            FALSE
## sqft_living15   1.081761  4.058068e+00   FALSE FALSE            FALSE
## bathrooms       1.395971  1.681520e-01   FALSE FALSE            FALSE
## view           20.557545  2.802533e-02   FALSE  TRUE            FALSE
## sqft_basement  59.697802  1.647890e+00   FALSE  TRUE            FALSE
## bedrooms        1.432047  7.286587e-02   FALSE FALSE            FALSE
## lat             1.066667  2.726865e+01   FALSE FALSE            FALSE
## waterfront    126.435714  1.121013e-02   FALSE  TRUE            FALSE
## floors          1.286028  3.363040e-02   FALSE FALSE            FALSE
## yr_renovated  224.842105  3.923547e-01   FALSE  TRUE            FALSE
## sqft_lot        1.219409  4.819797e+01   FALSE FALSE            FALSE
## sqft_lot15      1.149007  4.334398e+01   FALSE FALSE            FALSE
## yr_built        1.322835  6.501878e-01   FALSE FALSE            FALSE
## condition       2.480522  2.802533e-02   FALSE FALSE            FALSE
## .pos            1.000000  1.000000e+02   FALSE FALSE            FALSE
## long            1.032609  4.091699e+00   FALSE FALSE            FALSE
## .rnorm          1.000000  1.000000e+02   FALSE FALSE            FALSE
## id              1.000000  9.934981e+01   FALSE FALSE            FALSE
## zipcode         1.004090  3.923547e-01   FALSE FALSE            FALSE
## .category       0.000000  5.605067e-03    TRUE  TRUE               NA
```

```
## Warning in myplot_scatter(plt_feats_df, "percentUnique", "freqRatio",
## colorcol_name = "nzv", : converting nzv to class:factor
```

```
## Warning: Removed 6 rows containing missing values (geom_point).
```

```
## Warning: Removed 6 rows containing missing values (geom_point).
```

```
## Warning: Removed 6 rows containing missing values (geom_point).
```

![](Houses_tmplt_files/figure-html/select.features-1.png) 

```
##                   cor.y exclude.as.feat cor.y.abs cor.high.X freqRatio
## view          0.4069395               1 0.4069395         NA  20.55754
## sqft_basement 0.3317863               1 0.3317863         NA  59.69780
## waterfront    0.2782247               1 0.2782247         NA 126.43571
## yr_renovated  0.1227210               1 0.1227210         NA 224.84211
## .category            NA               1        NA         NA   0.00000
##               percentUnique zeroVar  nzv is.cor.y.abs.low
## view            0.028025335   FALSE TRUE            FALSE
## sqft_basement   1.647889692   FALSE TRUE            FALSE
## waterfront      0.011210134   FALSE TRUE            FALSE
## yr_renovated    0.392354689   FALSE TRUE            FALSE
## .category       0.005605067    TRUE TRUE               NA
```

![](Houses_tmplt_files/figure-html/select.features-2.png) 

```
## [1] "numeric data missing in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ 0s in glbObsAll: "
##      bedrooms     bathrooms    waterfront          view sqft_basement 
##            13            10         21450         19489         13126 
##  yr_renovated 
##         20699 
## [1] "numeric data w/ Infs in glbObsAll: "
## named integer(0)
## [1] "numeric data w/ NaNs in glbObsAll: "
## named integer(0)
## [1] "string data missing in glbObsAll: "
##    date id.date    .lcn 
##       0       0       0
```

```
## [1] "glb_feats_df:"
```

```
## [1] 22 12
```

```
##          id exclude.as.feat rsp_var
## price price            TRUE    TRUE
```

```
##          id cor.y exclude.as.feat cor.y.abs cor.high.X freqRatio
## price price    NA            TRUE        NA         NA        NA
##       percentUnique zeroVar nzv is.cor.y.abs.low interaction.feat
## price            NA      NA  NA               NA               NA
##       shapiro.test.p.value rsp_var_raw id_var rsp_var
## price                   NA          NA     NA    TRUE
```

```
## [1] "glb_feats_df vs. glbObsAll: "
```

```
## character(0)
```

```
## [1] "glbObsAll vs. glb_feats_df: "
```

```
## character(0)
```

```
##              label step_major step_minor label_minor    bgn  end elapsed
## 14 select.features          7          0           0 31.311 34.9    3.59
## 15      fit.models          8          0           0 34.901   NA      NA
```

## Step `8.0: fit models`

```r
fit.models_0_chunk_df <- myadd_chunk(NULL, "fit.models_0_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_0_bgn          1          0       setup 35.389  NA      NA
```

```r
# load(paste0(glb_out_pfx, "dsk.RData"))

get_model_sel_frmla <- function() {
    model_evl_terms <- c(NULL)
    # min.aic.fit might not be avl
    lclMdlEvlCriteria <- 
        glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)]
    for (metric in lclMdlEvlCriteria)
        model_evl_terms <- c(model_evl_terms, 
                             ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
    if (glb_is_classification && glb_is_binomial)
        model_evl_terms <- c(model_evl_terms, "-", "opt.prob.threshold.OOB")
    model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse = " "))
    return(model_sel_frmla)
}

get_dsp_models_df <- function() {
    dsp_models_cols <- c("id", 
                    glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                    grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
    dsp_models_df <- 
        #orderBy(get_model_sel_frmla(), glb_models_df)[, c("id", glbMdlMetricsEval)]
        orderBy(get_model_sel_frmla(), glb_models_df)[, dsp_models_cols]    
    nCvMdl <- sapply(glb_models_lst, function(mdl) nrow(mdl$results))
    nParams <- sapply(glb_models_lst, function(mdl) ifelse(mdl$method == "custom", 0, 
        nrow(subset(modelLookup(mdl$method), parameter != "parameter"))))
    
#     nCvMdl <- nCvMdl[names(nCvMdl) != "avNNet"]
#     nParams <- nParams[names(nParams) != "avNNet"]    
    
    if (length(cvMdlProblems <- nCvMdl[nCvMdl <= nParams]) > 0) {
        print("Cross Validation issues:")
        warning("Cross Validation issues:")        
        print(cvMdlProblems)
    }
    
    pltMdls <- setdiff(names(nCvMdl), names(cvMdlProblems))
    pltMdls <- setdiff(pltMdls, names(nParams[nParams == 0]))
    
    # length(pltMdls) == 21
    png(paste0(glb_out_pfx, "bestTune.png"), width = 480 * 2, height = 480 * 4)
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(ceiling(length(pltMdls) / 2.0), 2)))
    pltIx <- 1
    for (mdlId in pltMdls) {
        print(ggplot(glb_models_lst[[mdlId]], highBestTune = TRUE) + labs(title = mdlId),   
              vp = viewport(layout.pos.row = ceiling(pltIx / 2.0), 
                            layout.pos.col = ((pltIx - 1) %% 2) + 1))  
        pltIx <- pltIx + 1
    }
    dev.off()

    if (all(row.names(dsp_models_df) != dsp_models_df$id))
        row.names(dsp_models_df) <- dsp_models_df$id
    return(dsp_models_df)
}
#get_dsp_models_df()

if (glb_is_classification && glb_is_binomial && 
        (length(unique(glbObsFit[, glb_rsp_var])) < 2))
    stop("glbObsFit$", glb_rsp_var, ": contains less than 2 unique values: ",
         paste0(unique(glbObsFit[, glb_rsp_var]), collapse=", "))

max_cor_y_x_vars <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & !nzv & !is.cor.y.abs.low & 
                                is.na(cor.high.X)))[1:2, "id"]
max_cor_y_x_vars <- max_cor_y_x_vars[!is.na(max_cor_y_x_vars)]

if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_vars[1] != glb_Baseline_mdl_var) & 
        (glb_feats_df[glb_feats_df$id == max_cor_y_x_vars[1], "cor.y.abs"] > 
         glb_feats_df[glb_feats_df$id == glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_vars[1], " has a higher correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Model specs
c("id.prefix", "method", "type",
  # trainControl params
  "preProc.method", "cv.n.folds", "cv.n.repeats", "summary.fn",
  # train params
  "metric", "metric.maximize", "tune.df")
```

```
##  [1] "id.prefix"       "method"          "type"           
##  [4] "preProc.method"  "cv.n.folds"      "cv.n.repeats"   
##  [7] "summary.fn"      "metric"          "metric.maximize"
## [10] "tune.df"
```

```r
# Baseline
if (!is.null(glb_Baseline_mdl_var)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                            paste0("fit.models_0_", "Baseline"), major.inc = FALSE,
                                    label.minor = "mybaseln_classfr")
    ret_lst <- myfit_mdl(mdl_id="Baseline", 
                         model_method="mybaseln_classfr",
                        indep_vars_vctr=glb_Baseline_mdl_var,
                        rsp_var=glb_rsp_var,
                        fit_df=glbObsFit, OOB_df=glbObsOOB)
}    

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
if (glb_is_classification) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                                paste0("fit.models_0_", "MFO"), major.inc = FALSE,
                                        label.minor = "myMFO_classfr")
    
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "MFO", type = glb_model_type, trainControl.method = "none",
        train.method = ifelse(glb_is_regression, "lm", "myMFO_classfr"))),
                            indep_vars = ".rnorm", rsp_var = glb_rsp_var,
                            fit_df = glbObsFit, OOB_df = glbObsOOB)
    
        # "random" model - only for classification; 
        #   none needed for regression since it is same as MFO
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                                paste0("fit.models_0_", "Random"), major.inc = FALSE,
                                        label.minor = "myrandom_classfr")

#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)    
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Random", type = glb_model_type, trainControl.method = "none",
        train.method = "myrandom_classfr")),
                        indep_vars = ".rnorm", rsp_var = glb_rsp_var,
                        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Max.cor.Y.rcv.*X*"), major.inc = FALSE,
                                    label.minor = "glmnet")
```

```
##                            label step_major step_minor label_minor    bgn
## 1               fit.models_0_bgn          1          0       setup 35.389
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet 35.424
##      end elapsed
## 1 35.423   0.034
## 2     NA      NA
```

```r
ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
    id.prefix="Max.cor.Y.rcv.1X1", type=glb_model_type, trainControl.method="none",
    train.method="glmnet")),
                    indep_vars=max_cor_y_x_vars, rsp_var=glb_rsp_var, 
                    fit_df=glbObsFit, OOB_df=glbObsOOB)
```

```
## [1] "fitting model: Max.cor.Y.rcv.1X1###glmnet"
## [1] "    indep_vars: .pos,.rnorm"
```

```
## Loading required package: glmnet
## Loading required package: Matrix
## Loaded glmnet 2.0-2
```

```
## Fitting alpha = 0.1, lambda = 214 on full training set
```

![](Houses_tmplt_files/figure-html/fit.models_0-1.png) 

```
##             Length Class      Mode     
## a0           60    -none-     numeric  
## beta        120    dgCMatrix  S4       
## df           60    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       60    -none-     numeric  
## dev.ratio    60    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
## 526632.911885      2.059212  -2697.858937 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                           id       feats max.nTuningRuns
## 1 Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      0.728                 0.014 0.0008517554
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     378150.3     0.0007397307  -0.01796882       311717        -0.018509
```

```r
if (glbMdlCheckRcv) {
    # rcv_n_folds == 1 & rcv_n_repeats > 1 crashes
    for (rcv_n_folds in seq(3, glb_rcv_n_folds + 2, 2))
        for (rcv_n_repeats in seq(1, glb_rcv_n_repeats + 2, 2)) {
            
            # Experiment specific code to avoid caret crash
    #         lcl_tune_models_df <- rbind(data.frame()
    #                             ,data.frame(method = "glmnet", parameter = "alpha", 
    #                                         vals = "0.100 0.325 0.550 0.775 1.000")
    #                             ,data.frame(method = "glmnet", parameter = "lambda",
    #                                         vals = "9.342e-02")    
    #                                     )
            
            ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst =
                list(
                id.prefix = paste0("Max.cor.Y.rcv.", rcv_n_folds, "X", rcv_n_repeats), 
                type = glb_model_type, 
    # tune.df = lcl_tune_models_df,            
                trainControl.method = "repeatedcv",
                trainControl.number = rcv_n_folds, 
                trainControl.repeats = rcv_n_repeats,
                trainControl.classProbs = glb_is_classification,
                trainControl.summaryFunction = glbMdlMetricSummaryFn,
                train.method = "glmnet", train.metric = glbMdlMetricSummary, 
                train.maximize = glbMdlMetricMaximize)),
                                indep_vars = max_cor_y_x_vars, rsp_var = glb_rsp_var, 
                                fit_df = glbObsFit, OOB_df = glbObsOOB)
        }
    # Add parallel coordinates graph of glb_models_df[, glbMdlMetricsEval] to evaluate cv parameters
    tmp_models_cols <- c("id", "max.nTuningRuns",
                        glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                        grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
    print(myplot_parcoord(obs_df = subset(glb_models_df, 
                                          grepl("Max.cor.Y.rcv.", id, fixed = TRUE), 
                                            select = -feats)[, tmp_models_cols],
                          id_var = "id"))
}
        
# Useful for stacking decisions
# fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
#                     paste0("fit.models_0_", "Max.cor.Y[rcv.1X1.cp.0|]"), major.inc = FALSE,
#                                     label.minor = "rpart")
# 
# ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
#     id.prefix = "Max.cor.Y.rcv.1X1.cp.0", type = glb_model_type, trainControl.method = "none",
#     train.method = "rpart",
#     tune.df=data.frame(method="rpart", parameter="cp", min=0.0, max=0.0, by=0.1))),
#                     indep_vars=max_cor_y_x_vars, rsp_var=glb_rsp_var, 
#                     fit_df=glbObsFit, OOB_df=glbObsOOB)

#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)
# if (glb_is_regression || glb_is_binomial) # For multinomials this model will be run next by default
ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                        id.prefix = "Max.cor.Y", 
                        type = glb_model_type, trainControl.method = "repeatedcv",
                        trainControl.number = glb_rcv_n_folds, 
                        trainControl.repeats = glb_rcv_n_repeats,
                        trainControl.classProbs = glb_is_classification,
                        trainControl.summaryFunction = glbMdlMetricSummaryFn,
                        trainControl.allowParallel = glbMdlAllowParallel,                        
                        train.metric = glbMdlMetricSummary, 
                        train.maximize = glbMdlMetricMaximize,    
                        train.method = "rpart")),
                    indep_vars = max_cor_y_x_vars, rsp_var = glb_rsp_var, 
                    fit_df = glbObsFit, OOB_df = glbObsOOB)
```

```
## [1] "fitting model: Max.cor.Y##rcv#rpart"
## [1] "    indep_vars: .pos,.rnorm"
```

```
## Loading required package: rpart
```

```
## + Fold1.Rep1: cp=0.000591 
## - Fold1.Rep1: cp=0.000591 
## + Fold2.Rep1: cp=0.000591 
## - Fold2.Rep1: cp=0.000591 
## + Fold3.Rep1: cp=0.000591 
## - Fold3.Rep1: cp=0.000591 
## + Fold1.Rep2: cp=0.000591 
## - Fold1.Rep2: cp=0.000591 
## + Fold2.Rep2: cp=0.000591 
## - Fold2.Rep2: cp=0.000591 
## + Fold3.Rep2: cp=0.000591 
## - Fold3.Rep2: cp=0.000591 
## + Fold1.Rep3: cp=0.000591 
## - Fold1.Rep3: cp=0.000591 
## + Fold2.Rep3: cp=0.000591 
## - Fold2.Rep3: cp=0.000591 
## + Fold3.Rep3: cp=0.000591 
## - Fold3.Rep3: cp=0.000591
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00259 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Max.cor.Y", : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

```
## Loading required package: rpart.plot
```

![](Houses_tmplt_files/figure-html/fit.models_0-2.png) ![](Houses_tmplt_files/figure-html/fit.models_0-3.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 17841 
## 
##            CP nsplit rel error
## 1 0.002594806      0         1
## 
## Node number 1: 17841 observations
##   mean=545016.4, MSE=1.431195e+11 
## 
## n= 17841 
## 
## node), split, n, deviance, yval
##       * denotes terminal node
## 
## 1) root 17841 2.553395e+15 545016.4 *
##                     id       feats max.nTuningRuns
## 1 Max.cor.Y##rcv#rpart .pos,.rnorm               5
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                      4.583                 0.154            0
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1       378063               NA            0     308953.6               NA
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1      0.001561815       10732.82       0.0006249493
```

```r
if ((length(glbFeatsDateTime) > 0) && 
    (sum(grepl(paste(names(glbFeatsDateTime), "\\.day\\.minutes\\.poly\\.", sep = ""),
               names(glbObsAll))) > 0)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Max.cor.Y.Time.Poly"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars, 
            grep(paste(names(glbFeatsDateTime), "\\.day\\.minutes\\.poly\\.", sep = ""),
                        names(glbObsAll), value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
            id.prefix = "Max.cor.Y.Time.Poly", 
            type = glb_model_type, trainControl.method = "repeatedcv",
            trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

if ((length(glbFeatsDateTime) > 0) && 
    (sum(grepl(paste(names(glbFeatsDateTime), "\\.last[[:digit:]]", sep = ""),
               names(glbObsAll))) > 0)) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Max.cor.Y.Time.Lag"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars, 
            grep(paste(names(glbFeatsDateTime), "\\.last[[:digit:]]", sep = ""),
                        names(glbObsAll), value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Time.Lag", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

if (length(glbFeatsText) > 0) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Txt.*"), major.inc = FALSE,
                                    label.minor = "glmnet")

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.(?!([T|P]\\.))", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.nonTP", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        trainControl.allowParallel = glbMdlAllowParallel,                                
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.T\\.", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.onlyT", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)

    indepVars <- c(max_cor_y_x_vars)
    for (txtFeat in names(glbFeatsText))
        indepVars <- union(indepVars, 
            grep(paste(str_to_upper(substr(txtFeat, 1, 1)), "\\.P\\.", sep = ""),
                        names(glbObsAll), perl = TRUE, value = TRUE))
    indepVars <- myadjust_interaction_feats(indepVars)
    ret_lst <- myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
        id.prefix = "Max.cor.Y.Text.onlyP", 
        type = glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method = "repeatedcv",
        trainControl.number = glb_rcv_n_folds, trainControl.repeats = glb_rcv_n_repeats,
        trainControl.classProbs = glb_is_classification,
        trainControl.summaryFunction = glbMdlMetricSummaryFn,
        trainControl.allowParallel = glbMdlAllowParallel,        
        train.metric = glbMdlMetricSummary, 
        train.maximize = glbMdlMetricMaximize,    
        train.method = "glmnet")),
        indep_vars = indepVars,
        rsp_var = glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
}

# Interactions.High.cor.Y
if (length(int_feats <- setdiff(setdiff(unique(glb_feats_df$cor.high.X), NA), 
                                subset(glb_feats_df, nzv)$id)) > 0) {
    fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                    paste0("fit.models_0_", "Interact.High.cor.Y"), major.inc = FALSE,
                                    label.minor = "glmnet")

    ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
        id.prefix="Interact.High.cor.Y", 
        type=glb_model_type, trainControl.method="repeatedcv",
        trainControl.number=glb_rcv_n_folds, trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
        train.method="glmnet")),
        indep_vars=c(max_cor_y_x_vars, paste(max_cor_y_x_vars[1], int_feats, sep=":")),
        rsp_var=glb_rsp_var, 
        fit_df=glbObsFit, OOB_df=glbObsOOB)
}    

# Low.cor.X
fit.models_0_chunk_df <- myadd_chunk(fit.models_0_chunk_df, 
                        paste0("fit.models_0_", "Low.cor.X"), major.inc = FALSE,
                                     label.minor = "glmnet")
```

```
##                            label step_major step_minor label_minor    bgn
## 2 fit.models_0_Max.cor.Y.rcv.*X*          1          1      glmnet 35.424
## 3         fit.models_0_Low.cor.X          1          2      glmnet 43.186
##      end elapsed
## 2 43.185   7.761
## 3     NA      NA
```

```r
indep_vars <- subset(glb_feats_df, is.na(cor.high.X) & !nzv & 
                              (exclude.as.feat != 1))[, "id"]  
indep_vars <- myadjust_interaction_feats(indep_vars)
ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
        id.prefix="Low.cor.X", 
        type=glb_model_type, 
        tune.df = glbMdlTuneParams,        
        trainControl.method="repeatedcv",
        trainControl.number=glb_rcv_n_folds, trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
        train.method="glmnet")),
        indep_vars=indep_vars, rsp_var=glb_rsp_var, 
        fit_df = glbObsFit, OOB_df = glbObsOOB)
```

```
## [1] "fitting model: Low.cor.X##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 214 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = "Low.cor.X", : model's bestTune found at an extreme of
## tuneGrid for parameter: alpha
```

![](Houses_tmplt_files/figure-html/fit.models_0-4.png) ![](Houses_tmplt_files/figure-html/fit.models_0-5.png) 

```
##             Length Class      Mode     
## a0           60    -none-     numeric  
## beta        120    dgCMatrix  S4       
## df           60    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       60    -none-     numeric  
## dev.ratio    60    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
## 526632.911885      2.059212  -2697.858937 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                      id       feats max.nTuningRuns
## 1 Low.cor.X##rcv#glmnet .pos,.rnorm              20
##   min.elapsedtime.everything min.elapsedtime.final max.R.sq.fit
## 1                       4.91                 0.009 0.0008517554
##   min.RMSE.fit max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     378055.2     0.0007397307  -0.01796882       311717        -0.018509
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1     0.0009340522       10688.73       0.0006175565
```

```r
fit.models_0_chunk_df <- 
    myadd_chunk(fit.models_0_chunk_df, "fit.models_0_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                    label step_major step_minor label_minor    bgn   end
## 3 fit.models_0_Low.cor.X          1          2      glmnet 43.186 49.68
## 4       fit.models_0_end          1          3    teardown 49.681    NA
##   elapsed
## 3   6.494
## 4      NA
```

```r
rm(ret_lst)

glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor    bgn    end elapsed
## 15 fit.models          8          0           0 34.901 49.691   14.79
## 16 fit.models          8          1           1 49.692     NA      NA
```


```r
fit.models_1_chunk_df <- myadd_chunk(NULL, "fit.models_1_bgn", label.minor="setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_1_bgn          1          0       setup 50.621  NA      NA
```

```r
#stop(here"); glb2Sav(); all.equal(glb_models_df, sav_models_df)
topindep_var <- NULL; interact_vars <- NULL;
for (mdl_id_pfx in names(glbMdlFamilies)) {
    fit.models_1_chunk_df <- 
        myadd_chunk(fit.models_1_chunk_df, paste0("fit.models_1_", mdl_id_pfx),
                    major.inc = FALSE, label.minor = "setup")

    indep_vars <- NULL;

    if (grepl("\\.Interact", mdl_id_pfx)) {
        if (is.null(topindep_var) && is.null(interact_vars)) {
        #   select best glmnet model upto now
            dsp_models_df <- orderBy(model_sel_frmla <- get_model_sel_frmla(),
                                     glb_models_df)
            dsp_models_df <- subset(dsp_models_df, 
                                    grepl(".glmnet", id, fixed = TRUE))
            bst_mdl_id <- dsp_models_df$id[1]
            mdl_id_pfx <- 
                paste(c(head(unlist(strsplit(bst_mdl_id, "[.]")), -1), "Interact"),
                      collapse=".")
        #   select important features
            if (is.null(bst_featsimp_df <- 
                        myget_feats_importance(glb_models_lst[[bst_mdl_id]]))) {
                warning("Base model for RFE.Interact: ", bst_mdl_id, 
                        " has no important features")
                next
            }    
            
            topindep_ix <- 1
            while (is.null(topindep_var) && (topindep_ix <= nrow(bst_featsimp_df))) {
                topindep_var <- row.names(bst_featsimp_df)[topindep_ix]
                if (grepl(".fctr", topindep_var, fixed=TRUE))
                    topindep_var <- 
                        paste0(unlist(strsplit(topindep_var, ".fctr"))[1], ".fctr")
                if (topindep_var %in% names(glbFeatsInteractionOnly)) {
                    topindep_var <- NULL; topindep_ix <- topindep_ix + 1
                } else break
            }
            
        #   select features with importance > max(10, importance of .rnorm) & is not highest
        #       combine factor dummy features to just the factor feature
            if (length(pos_rnorm <- 
                       grep(".rnorm", row.names(bst_featsimp_df), fixed=TRUE)) > 0)
                imp_rnorm <- bst_featsimp_df[pos_rnorm, 1] else
                imp_rnorm <- NA    
            imp_cutoff <- max(10, imp_rnorm, na.rm=TRUE)
            interact_vars <- 
                tail(row.names(subset(bst_featsimp_df, 
                                      imp > imp_cutoff)), -1)
            if (length(interact_vars) > 0) {
                interact_vars <-
                    myadjust_interaction_feats(myextract_actual_feats(interact_vars))
                interact_vars <- 
                    interact_vars[!grepl(topindep_var, interact_vars, fixed=TRUE)]
            }
            ### bid0_sp only
#             interact_vars <- c(
#     "biddable", "D.ratio.sum.TfIdf.wrds.n", "D.TfIdf.sum.stem.stop.Ratio", "D.sum.TfIdf",
#     "D.TfIdf.sum.post.stop", "D.TfIdf.sum.post.stem", "D.ratio.wrds.stop.n.wrds.n", "D.chrs.uppr.n.log",
#     "D.chrs.n.log", "color.fctr"
#     # , "condition.fctr", "prdl.my.descr.fctr"
#                                 )
#            interact_vars <- setdiff(interact_vars, c("startprice.dgt2.is9", "color.fctr"))
            ###
            indep_vars <- myextract_actual_feats(row.names(bst_featsimp_df))
            indep_vars <- setdiff(indep_vars, topindep_var)
            if (length(interact_vars) > 0) {
                indep_vars <- 
                    setdiff(indep_vars, myextract_actual_feats(interact_vars))
                indep_vars <- c(indep_vars, 
                    paste(topindep_var, setdiff(interact_vars, topindep_var), 
                          sep = "*"))
            } else indep_vars <- union(indep_vars, topindep_var)
        }
    }
    
    if (is.null(indep_vars))
        indep_vars <- glb_mdl_feats_lst[[mdl_id_pfx]]

    if (is.null(indep_vars) && grepl("RFE\\.", mdl_id_pfx))
        indep_vars <- myextract_actual_feats(predictors(rfe_fit_results))
    
    if (is.null(indep_vars))
        indep_vars <- subset(glb_feats_df, !nzv & (exclude.as.feat != 1))[, "id"]
    
    if ((length(indep_vars) == 1) && (grepl("^%<d-%", indep_vars))) {    
        indep_vars <- 
            eval(parse(text = str_trim(unlist(strsplit(indep_vars, "%<d-%"))[2])))
    }    

    indep_vars <- myadjust_interaction_feats(indep_vars)
    
    if (grepl("\\.Interact", mdl_id_pfx)) { 
        # if (method != tail(unlist(strsplit(bst_mdl_id, "[.]")), 1)) next
        if (is.null(glbMdlFamilies[[mdl_id_pfx]])) {
            if (!is.null(glbMdlFamilies[["Best.Interact"]]))
                glbMdlFamilies[[mdl_id_pfx]] <-
                    glbMdlFamilies[["Best.Interact"]]
        }
    }
    
    if (!is.null(glbObsFitOutliers[[mdl_id_pfx]])) {
        fitobs_df <- glbObsFit[!(glbObsFit[, glbFeatsId] %in%
                                         glbObsFitOutliers[[mdl_id_pfx]]), ]
    } else fitobs_df <- glbObsFit

    if (is.null(glbMdlFamilies[[mdl_id_pfx]]))
        mdl_methods <- glbMdlMethods else
        mdl_methods <- glbMdlFamilies[[mdl_id_pfx]]    

    for (method in mdl_methods) {
        if (method %in% c("rpart", "rf")) {
            # rpart:    fubar's the tree
            # rf:       skip the scenario w/ .rnorm for speed
            indep_vars <- setdiff(indep_vars, c(".rnorm"))
            #mdl_id <- paste0(mdl_id_pfx, ".no.rnorm")
        } 

        fit.models_1_chunk_df <- myadd_chunk(fit.models_1_chunk_df, 
                            paste0("fit.models_1_", mdl_id_pfx), major.inc = FALSE,
                                    label.minor = method)

        ret_lst <- 
            myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
            id.prefix = mdl_id_pfx, 
            type = glb_model_type, 
            tune.df = glbMdlTuneParams,
            trainControl.method = "repeatedcv", # or "none" if nominalWorkflow is crashing
            trainControl.number = glb_rcv_n_folds,
            trainControl.repeats = glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            trainControl.allowParallel = glbMdlAllowParallel,            
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method = method)),
            indep_vars = indep_vars, rsp_var = glb_rsp_var, 
            fit_df = fitobs_df, OOB_df = glbObsOOB)
        
#         ntv_mdl <- glmnet(x = as.matrix(
#                               fitobs_df[, indep_vars]), 
#                           y = as.factor(as.character(
#                               fitobs_df[, glb_rsp_var])),
#                           family = "multinomial")
#         bgn = 1; end = 100;
#         ntv_mdl <- glmnet(x = as.matrix(
#                               subset(fitobs_df, pop.fctr != "crypto")[bgn:end, indep_vars]), 
#                           y = as.factor(as.character(
#                               subset(fitobs_df, pop.fctr != "crypto")[bgn:end, glb_rsp_var])),
#                           family = "multinomial")
    }
}
```

```
##                label step_major step_minor label_minor    bgn    end
## 1   fit.models_1_bgn          1          0       setup 50.621 50.631
## 2 fit.models_1_All.X          1          1       setup 50.632     NA
##   elapsed
## 1    0.01
## 2      NA
##                label step_major step_minor label_minor    bgn    end
## 2 fit.models_1_All.X          1          1       setup 50.632 50.638
## 3 fit.models_1_All.X          1          2      glmnet 50.638     NA
##   elapsed
## 2   0.006
## 3      NA
## [1] "fitting model: All.X##rcv#glmnet"
## [1] "    indep_vars: .pos,.rnorm"
## Aggregating results
## Selecting tuning parameters
## Fitting alpha = 0.1, lambda = 214 on full training set
```

```
## Warning in myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst
## = list(id.prefix = mdl_id_pfx, : model's bestTune found at an extreme of
## tuneGrid for parameter: alpha
```

![](Houses_tmplt_files/figure-html/fit.models_1-1.png) ![](Houses_tmplt_files/figure-html/fit.models_1-2.png) 

```
##             Length Class      Mode     
## a0           60    -none-     numeric  
## beta        120    dgCMatrix  S4       
## df           60    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       60    -none-     numeric  
## dev.ratio    60    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
## 526632.911885      2.059212  -2697.858937 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"     
##                  id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glmnet .pos,.rnorm              20                      3.675
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit max.Adj.R.sq.fit
## 1                 0.009 0.0008517554     378055.2     0.0007397307
##   max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## 1  -0.01796882       311717        -0.018509     0.0009340522
##   min.RMSESD.fit max.RsquaredSD.fit
## 1       10688.73       0.0006175565
##                label step_major step_minor label_minor    bgn    end
## 3 fit.models_1_All.X          1          2      glmnet 50.638 55.948
## 4 fit.models_1_All.X          1          3         glm 55.948     NA
##   elapsed
## 3    5.31
## 4      NA
## [1] "fitting model: All.X##rcv#glm"
## [1] "    indep_vars: .pos,.rnorm"
## + Fold1.Rep1: parameter=none 
## - Fold1.Rep1: parameter=none 
## + Fold2.Rep1: parameter=none 
## - Fold2.Rep1: parameter=none 
## + Fold3.Rep1: parameter=none 
## - Fold3.Rep1: parameter=none 
## + Fold1.Rep2: parameter=none 
## - Fold1.Rep2: parameter=none 
## + Fold2.Rep2: parameter=none 
## - Fold2.Rep2: parameter=none 
## + Fold3.Rep2: parameter=none 
## - Fold3.Rep2: parameter=none 
## + Fold1.Rep3: parameter=none 
## - Fold1.Rep3: parameter=none 
## + Fold2.Rep3: parameter=none 
## - Fold2.Rep3: parameter=none 
## + Fold3.Rep3: parameter=none 
## - Fold3.Rep3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Houses_tmplt_files/figure-html/fit.models_1-3.png) ![](Houses_tmplt_files/figure-html/fit.models_1-4.png) ![](Houses_tmplt_files/figure-html/fit.models_1-5.png) ![](Houses_tmplt_files/figure-html/fit.models_1-6.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -475209  -223568   -92033   102663  7162276  
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  5.265e+05  5.663e+03  92.975  < 2e-16 ***
## .pos         2.070e+00  5.498e-01   3.765 0.000167 ***
## .rnorm      -2.744e+03  2.851e+03  -0.963 0.335747    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for gaussian family taken to be 143021665226)
## 
##     Null deviance: 2.5534e+15  on 17840  degrees of freedom
## Residual deviance: 2.5512e+15  on 17838  degrees of freedom
## AIC: 508904
## 
## Number of Fisher Scoring iterations: 2
## 
##               id       feats max.nTuningRuns min.elapsedtime.everything
## 1 All.X##rcv#glm .pos,.rnorm               1                      1.536
##   min.elapsedtime.final max.R.sq.fit min.RMSE.fit min.aic.fit
## 1                 0.034 0.0008517918     378055.4    508904.2
##   max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB max.Adj.R.sq.OOB
## 1     0.0007397671  -0.01808814     311735.3      -0.01862838
##   max.Rsquared.fit min.RMSESD.fit max.RsquaredSD.fit
## 1     0.0009346427       10688.51       0.0006183888
```

```r
# Check if other preProcess methods improve model performance
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_preProc", major.inc = FALSE,
                label.minor = "preProc")
```

```
##                  label step_major step_minor label_minor    bgn    end
## 4   fit.models_1_All.X          1          3         glm 55.948 61.599
## 5 fit.models_1_preProc          1          4     preProc 61.599     NA
##   elapsed
## 4   5.651
## 5      NA
```

```r
mdl_id <- orderBy(get_model_sel_frmla(), glb_models_df)[1, "id"]
indep_vars_vctr <- trim(unlist(strsplit(glb_models_df[glb_models_df$id == mdl_id,
                                                      "feats"], "[,]")))
method <- tail(unlist(strsplit(mdl_id, "[.]")), 1)
mdl_id_pfx <- paste0(head(unlist(strsplit(mdl_id, "[.]")), -1), collapse = ".")
if (!is.null(glbObsFitOutliers[[mdl_id_pfx]])) {
    fitobs_df <- glbObsFit[!(glbObsFit[, glbFeatsId] %in%
                                     glbObsFitOutliers[[mdl_id_pfx]]), ]
} else fitobs_df <- glbObsFit

for (prePr in glb_preproc_methods) {   
    # The operations are applied in this order: 
    #   Box-Cox/Yeo-Johnson transformation, centering, scaling, range, imputation, PCA, ICA then spatial sign.
    
    ret_lst <- myfit_mdl(mdl_specs_lst=myinit_mdl_specs_lst(mdl_specs_lst=list(
            id.prefix=mdl_id_pfx, 
            type=glb_model_type, tune.df=glbMdlTuneParams,
            trainControl.method="repeatedcv",
            trainControl.number=glb_rcv_n_folds,
            trainControl.repeats=glb_rcv_n_repeats,
            trainControl.classProbs = glb_is_classification,
            trainControl.summaryFunction = glbMdlMetricSummaryFn,
            train.metric = glbMdlMetricSummary, 
            train.maximize = glbMdlMetricMaximize,    
            train.method=method, train.preProcess=prePr)),
            indep_vars=indep_vars_vctr, rsp_var=glb_rsp_var, 
            fit_df=fitobs_df, OOB_df=glbObsOOB)
}            
    
    # If (All|RFE).X.glm is less accurate than Low.Cor.X.glm
    #   check NA coefficients & filter appropriate terms in indep_vars_vctr
#     if (method == "glm") {
#         orig_glm <- glb_models_lst[[paste0(mdl_id, ".", model_method)]]$finalModel
#         orig_glm <- glb_models_lst[["All.X.glm"]]$finalModel; print(summary(orig_glm))
#         orig_glm <- glb_models_lst[["RFE.X.glm"]]$finalModel; print(summary(orig_glm))
#           require(car)
#           vif_orig_glm <- vif(orig_glm); print(vif_orig_glm)
#           # if vif errors out with "there are aliased coefficients in the model"
#               alias_orig_glm <- alias(orig_glm); alias_complete_orig_glm <- (alias_orig_glm$Complete > 0); alias_complete_orig_glm <- alias_complete_orig_glm[rowSums(alias_complete_orig_glm) > 0, colSums(alias_complete_orig_glm) > 0]; print(alias_complete_orig_glm)
#           print(vif_orig_glm[!is.na(vif_orig_glm) & (vif_orig_glm == Inf)])
#           print(which.max(vif_orig_glm))
#           print(sort(vif_orig_glm[vif_orig_glm >= 1.0e+03], decreasing=TRUE))
#           glbObsFit[c(1143, 3637, 3953, 4105), c("UniqueID", "Popular", "H.P.quandary", "Headline")]
#           glb_feats_df[glb_feats_df$id %in% grep("[HSA]\\.chrs.n.log", glb_feats_df$id, value=TRUE) | glb_feats_df$cor.high.X %in%    grep("[HSA]\\.chrs.n.log", glb_feats_df$id, value=TRUE), ]
#           all.equal(glbObsAll$S.chrs.uppr.n.log, glbObsAll$A.chrs.uppr.n.log)
#           cor(glbObsAll$S.T.herald, glbObsAll$S.T.tribun)
#           mydspObs(Abstract.contains="[Dd]iar", cols=("Abstract"), all=TRUE)
#           subset(glb_feats_df, cor.y.abs <= glb_feats_df[glb_feats_df$id == ".rnorm", "cor.y.abs"])
#         corxx_mtrx <- cor(data.matrix(glbObsAll[, setdiff(names(glbObsAll), myfind_chr_cols_df(glbObsAll))]), use="pairwise.complete.obs"); abs_corxx_mtrx <- abs(corxx_mtrx); diag(abs_corxx_mtrx) <- 0
#           which.max(abs_corxx_mtrx["S.T.tribun", ])
#           abs_corxx_mtrx["A.npnct08.log", "S.npnct08.log"]
#         step_glm <- step(orig_glm)
#     }
    # Since caret does not optimize rpart well
#     if (method == "rpart")
#         ret_lst <- myfit_mdl(mdl_id=paste0(mdl_id_pfx, ".cp.0"), model_method=method,
#                                 indep_vars_vctr=indep_vars_vctr,
#                                 model_type=glb_model_type,
#                                 rsp_var=glb_rsp_var,
#                                 fit_df=glbObsFit, OOB_df=glbObsOOB,        
#             n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))

# User specified
#   Ensure at least 2 vars in each regression; else varImp crashes
# sav_models_lst <- glb_models_lst; sav_models_df <- glb_models_df; sav_featsimp_df <- glb_featsimp_df; all.equal(sav_featsimp_df, glb_featsimp_df)
# glb_models_lst <- sav_models_lst; glb_models_df <- sav_models_df; glm_featsimp_df <- sav_featsimp_df

    # easier to exclude features
# require(gdata) # needed for trim
# mdl_id <- "";
# indep_vars_vctr <- head(subset(glb_models_df, grepl("All\\.X\\.", mdl_id), select=feats)
#                         , 1)[, "feats"]
# indep_vars_vctr <- trim(unlist(strsplit(indep_vars_vctr, "[,]")))
# indep_vars_vctr <- setdiff(indep_vars_vctr, ".rnorm")

    # easier to include features
#stop(here"); sav_models_df <- glb_models_df; glb_models_df <- sav_models_df
# !_sp
# mdl_id <- "csm"; indep_vars_vctr <- c(NULL
#     ,"prdline.my.fctr", "prdline.my.fctr:.clusterid.fctr"
#     ,"prdline.my.fctr*biddable"
#     #,"prdline.my.fctr*startprice.log"
#     #,"prdline.my.fctr*startprice.diff"    
#     ,"prdline.my.fctr*condition.fctr"
#     ,"prdline.my.fctr*D.terms.post.stop.n"
#     #,"prdline.my.fctr*D.terms.post.stem.n"
#     ,"prdline.my.fctr*cellular.fctr"    
# #    ,"<feat1>:<feat2>"
#                                            )
# for (method in glbMdlMethods) {
#     ret_lst <- myfit_mdl(mdl_id=mdl_id, model_method=method,
#                                 indep_vars_vctr=indep_vars_vctr,
#                                 model_type=glb_model_type,
#                                 rsp_var=glb_rsp_var,
#                                 fit_df=glbObsFit, OOB_df=glbObsOOB,
#                     n_cv_folds=glb_rcv_n_folds, tune_models_df=glbMdlTuneParams)
#     csm_mdl_id <- paste0(mdl_id, ".", method)
#     csm_featsimp_df <- myget_feats_importance(glb_models_lst[[paste0(mdl_id, ".",
#                                                                      method)]]);               print(head(csm_featsimp_df))
# }
###

# Ntv.1.lm <- lm(reformulate(indep_vars_vctr, glb_rsp_var), glbObsTrn); print(summary(Ntv.1.lm))

#glb_models_df[, "max.Accuracy.OOB", FALSE]
#varImp(glb_models_lst[["Low.cor.X.glm"]])
#orderBy(~ -Overall, varImp(glb_models_lst[["All.X.2.glm"]])$imp)
#orderBy(~ -Overall, varImp(glb_models_lst[["All.X.3.glm"]])$imp)
#glb_feats_df[grepl("npnct28", glb_feats_df$id), ]

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glbObsFit), 
#                          union(glb_rsp_var, glbFeatsExclude)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]
    
    # template for myfit_mdl
    #   rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #       only for OOB in trainControl ?
    
#     ret_lst <- myfit_mdl_fn(mdl_id=paste0(mdl_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var,
#                             fit_df=glbObsFit, OOB_df=glbObsOOB,
#                             n_cv_folds=glb_rcv_n_folds, tune_models_df=glbMdlTuneParams,
#                             model_loss_mtrx=glbMdlMetric_terms,
#                             model_summaryFunction=glbMdlMetricSummaryFn,
#                             model_metric=glbMdlMetricSummary,
#                             model_metric_maximize=glbMdlMetricMaximize)

# Simplify a model
# fit_df <- glbObsFit; glb_mdl <- step(<complex>_mdl)

# Non-caret models
#     rpart_area_mdl <- rpart(reformulate("Area", response=glb_rsp_var), 
#                                data=glbObsFit, #method="class", 
#                                control=rpart.control(cp=0.12),
#                            parms=list(loss=glbMdlMetric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 

print(glb_models_df)
```

```
##                                                    id       feats
## Max.cor.Y.rcv.1X1###glmnet Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm
## Max.cor.Y##rcv#rpart             Max.cor.Y##rcv#rpart .pos,.rnorm
## Low.cor.X##rcv#glmnet           Low.cor.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glmnet                   All.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glm                         All.X##rcv#glm .pos,.rnorm
##                            max.nTuningRuns min.elapsedtime.everything
## Max.cor.Y.rcv.1X1###glmnet               0                      0.728
## Max.cor.Y##rcv#rpart                     5                      4.583
## Low.cor.X##rcv#glmnet                   20                      4.910
## All.X##rcv#glmnet                       20                      3.675
## All.X##rcv#glm                           1                      1.536
##                            min.elapsedtime.final max.R.sq.fit min.RMSE.fit
## Max.cor.Y.rcv.1X1###glmnet                 0.014 0.0008517554     378150.3
## Max.cor.Y##rcv#rpart                       0.154 0.0000000000     378063.0
## Low.cor.X##rcv#glmnet                      0.009 0.0008517554     378055.2
## All.X##rcv#glmnet                          0.009 0.0008517554     378055.2
## All.X##rcv#glm                             0.034 0.0008517918     378055.4
##                            max.Adj.R.sq.fit max.R.sq.OOB min.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet     0.0007397307  -0.01796882     311717.0
## Max.cor.Y##rcv#rpart                     NA   0.00000000     308953.6
## Low.cor.X##rcv#glmnet          0.0007397307  -0.01796882     311717.0
## All.X##rcv#glmnet              0.0007397307  -0.01796882     311717.0
## All.X##rcv#glm                 0.0007397671  -0.01808814     311735.3
##                            max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet      -0.01850900               NA
## Max.cor.Y##rcv#rpart                     NA     0.0015618152
## Low.cor.X##rcv#glmnet           -0.01850900     0.0009340522
## All.X##rcv#glmnet               -0.01850900     0.0009340522
## All.X##rcv#glm                  -0.01862838     0.0009346427
##                            min.RMSESD.fit max.RsquaredSD.fit min.aic.fit
## Max.cor.Y.rcv.1X1###glmnet             NA                 NA          NA
## Max.cor.Y##rcv#rpart             10732.82       0.0006249493          NA
## Low.cor.X##rcv#glmnet            10688.73       0.0006175565          NA
## All.X##rcv#glmnet                10688.73       0.0006175565          NA
## All.X##rcv#glm                   10688.51       0.0006183888    508904.2
```

```r
rm(ret_lst)
fit.models_1_chunk_df <- 
    myadd_chunk(fit.models_1_chunk_df, "fit.models_1_end", major.inc = FALSE,
                label.minor = "teardown")
```

```
##                  label step_major step_minor label_minor    bgn    end
## 5 fit.models_1_preProc          1          4     preProc 61.599 61.695
## 6     fit.models_1_end          1          5    teardown 61.696     NA
##   elapsed
## 5   0.096
## 6      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc = FALSE)
```

```
##         label step_major step_minor label_minor    bgn    end elapsed
## 16 fit.models          8          1           1 49.692 61.708  12.017
## 17 fit.models          8          2           2 61.709     NA      NA
```


```r
fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "setup")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_2_bgn          1          0       setup 65.005  NA      NA
```

```r
plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                                                    id       feats
## Max.cor.Y.rcv.1X1###glmnet Max.cor.Y.rcv.1X1###glmnet .pos,.rnorm
## Max.cor.Y##rcv#rpart             Max.cor.Y##rcv#rpart .pos,.rnorm
## Low.cor.X##rcv#glmnet           Low.cor.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glmnet                   All.X##rcv#glmnet .pos,.rnorm
## All.X##rcv#glm                         All.X##rcv#glm .pos,.rnorm
##                            max.nTuningRuns max.R.sq.fit max.Adj.R.sq.fit
## Max.cor.Y.rcv.1X1###glmnet               0 0.0008517554     0.0007397307
## Max.cor.Y##rcv#rpart                     5 0.0000000000               NA
## Low.cor.X##rcv#glmnet                   20 0.0008517554     0.0007397307
## All.X##rcv#glmnet                       20 0.0008517554     0.0007397307
## All.X##rcv#glm                           1 0.0008517918     0.0007397671
##                            max.R.sq.OOB max.Adj.R.sq.OOB max.Rsquared.fit
## Max.cor.Y.rcv.1X1###glmnet  -0.01796882      -0.01850900               NA
## Max.cor.Y##rcv#rpart         0.00000000               NA     0.0015618152
## Low.cor.X##rcv#glmnet       -0.01796882      -0.01850900     0.0009340522
## All.X##rcv#glmnet           -0.01796882      -0.01850900     0.0009340522
## All.X##rcv#glm              -0.01808814      -0.01862838     0.0009346427
##                            inv.elapsedtime.everything
## Max.cor.Y.rcv.1X1###glmnet                  1.3736264
## Max.cor.Y##rcv#rpart                        0.2181977
## Low.cor.X##rcv#glmnet                       0.2036660
## All.X##rcv#glmnet                           0.2721088
## All.X##rcv#glm                              0.6510417
##                            inv.elapsedtime.final inv.RMSE.fit inv.RMSE.OOB
## Max.cor.Y.rcv.1X1###glmnet             71.428571 2.644451e-06 3.208038e-06
## Max.cor.Y##rcv#rpart                    6.493506 2.645062e-06 3.236732e-06
## Low.cor.X##rcv#glmnet                 111.111111 2.645116e-06 3.208038e-06
## All.X##rcv#glmnet                     111.111111 2.645116e-06 3.208038e-06
## All.X##rcv#glm                         29.411765 2.645115e-06 3.207850e-06
##                             inv.aic.fit
## Max.cor.Y.rcv.1X1###glmnet           NA
## Max.cor.Y##rcv#rpart                 NA
## Low.cor.X##rcv#glmnet                NA
## All.X##rcv#glmnet                    NA
## All.X##rcv#glm             1.965007e-06
```

```r
# print(myplot_radar(radar_inp_df=plt_models_df))
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(mdl_id %in% grep("random|MFO", plt_models_df$id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## [1] "var:min.RMSESD.fit"
## [1] "var:max.RsquaredSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "id", FALSE]
pltCI_models_df <- glb_models_df[, "id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), 
            paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- 
                unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data,
                                       sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

# mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("id", "model_method")], 
#                         all.x=TRUE)

png(paste0(glb_out_pfx, "models_bar.png"), width=480*3, height=480*2)
#print(gp <- myplot_bar(mltd_models_df, "id", "value", colorcol_name="model_method") + 
print(gp <- myplot_bar(df=mltd_models_df, xcol_name="id", ycol_names="value") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=mdl_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
```

```
## Warning: Removed 1 rows containing missing values (position_stack).
```

```
## Warning: Removed 4 rows containing missing values (geom_errorbar).
```

```r
dev.off()
```

```
## quartz_off_screen 
##                 2
```

```r
print(gp)
```

```
## Warning: Removed 1 rows containing missing values (position_stack).
```

```
## Warning: Removed 4 rows containing missing values (geom_errorbar).
```

![](Houses_tmplt_files/figure-html/fit.models_2-1.png) 

```r
dsp_models_cols <- c("id", 
                    glbMdlMetricsEval[glbMdlMetricsEval %in% names(glb_models_df)],
                    grep("opt.", names(glb_models_df), fixed = TRUE, value = TRUE)) 
# if (glb_is_classification && glb_is_binomial) 
#     dsp_models_cols <- c(dsp_models_cols, "opt.prob.threshold.OOB")
print(dsp_models_df <- orderBy(get_model_sel_frmla(), glb_models_df)[, dsp_models_cols])
```

```
##                           id min.RMSE.OOB max.R.sq.OOB max.Adj.R.sq.fit
## 2       Max.cor.Y##rcv#rpart     308953.6   0.00000000               NA
## 3      Low.cor.X##rcv#glmnet     311717.0  -0.01796882     0.0007397307
## 4          All.X##rcv#glmnet     311717.0  -0.01796882     0.0007397307
## 1 Max.cor.Y.rcv.1X1###glmnet     311717.0  -0.01796882     0.0007397307
## 5             All.X##rcv#glm     311735.3  -0.01808814     0.0007397671
##   min.RMSE.fit
## 2     378063.0
## 3     378055.2
## 4     378055.2
## 1     378150.3
## 5     378055.4
```

```r
# print(myplot_radar(radar_inp_df = dsp_models_df))
print("Metrics used for model selection:"); print(get_model_sel_frmla())
```

```
## [1] "Metrics used for model selection:"
```

```
## ~+min.RMSE.OOB - max.R.sq.OOB - max.Adj.R.sq.fit + min.RMSE.fit
## <environment: 0x7fae199e90e8>
```

```r
print(sprintf("Best model id: %s", dsp_models_df[1, "id"]))
```

```
## [1] "Best model id: Max.cor.Y##rcv#rpart"
```

```r
glb_get_predictions <- function(df, mdl_id, rsp_var, prob_threshold_def=NULL, verbose=FALSE) {
    mdl <- glb_models_lst[[mdl_id]]
    
    clmnNames <- mygetPredictIds(rsp_var, mdl_id)
    predct_var_name <- clmnNames$value        
    predct_prob_var_name <- clmnNames$prob
    predct_accurate_var_name <- clmnNames$is.acc
    predct_error_var_name <- clmnNames$err
    predct_erabs_var_name <- clmnNames$err.abs

    if (glb_is_regression) {
        df[, predct_var_name] <- predict(mdl, newdata=df, type="raw")
        if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_var_name) + 
                  facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="glm"))

        df[, predct_error_var_name] <- df[, predct_var_name] - df[, glb_rsp_var]
        if (verbose) print(myplot_scatter(df, predct_var_name, predct_error_var_name) + 
                  #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="auto"))
        if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_error_var_name) + 
                  #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
                  stat_smooth(method="glm"))
        
        df[, predct_erabs_var_name] <- abs(df[, predct_error_var_name])
        if (verbose) print(head(orderBy(reformulate(c("-", predct_erabs_var_name)), df)))
        
        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])
    }

    if (glb_is_classification && glb_is_binomial) {
        prob_threshold <- glb_models_df[glb_models_df$id == mdl_id, 
                                        "opt.prob.threshold.OOB"]
        if (is.null(prob_threshold) || is.na(prob_threshold)) {
            warning("Using default probability threshold: ", prob_threshold_def)
            if (is.null(prob_threshold <- prob_threshold_def))
                stop("Default probability threshold is NULL")
        }
        
        df[, predct_prob_var_name] <- predict(mdl, newdata = df, type = "prob")[, 2]
        df[, predct_var_name] <- 
        		factor(levels(df[, glb_rsp_var])[
    				(df[, predct_prob_var_name] >=
    					prob_threshold) * 1 + 1], levels(df[, glb_rsp_var]))
    
#         if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_var_name) + 
#                   facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="glm"))

        df[, predct_error_var_name] <- df[, predct_var_name] != df[, glb_rsp_var]
#         if (verbose) print(myplot_scatter(df, predct_var_name, predct_error_var_name) + 
#                   #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="auto"))
#         if (verbose) print(myplot_scatter(df, glb_rsp_var, predct_error_var_name) + 
#                   #facet_wrap(reformulate(glbFeatsCategory), scales = "free") + 
#                   stat_smooth(method="glm"))
        
        # if prediction is a TP (true +ve), measure distance from 1.0
        tp <- which((df[, predct_var_name] == df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[2]))
        df[tp, predct_erabs_var_name] <- abs(1 - df[tp, predct_prob_var_name])
        #rowIx <- which.max(df[tp, predct_erabs_var_name]); df[tp, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a TN (true -ve), measure distance from 0.0
        tn <- which((df[, predct_var_name] == df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[tn, predct_erabs_var_name] <- abs(0 - df[tn, predct_prob_var_name])
        #rowIx <- which.max(df[tn, predct_erabs_var_name]); df[tn, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FP (flse +ve), measure distance from 0.0
        fp <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[2]))
        df[fp, predct_erabs_var_name] <- abs(0 - df[fp, predct_prob_var_name])
        #rowIx <- which.max(df[fp, predct_erabs_var_name]); df[fp, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]
        
        # if prediction is a FN (flse -ve), measure distance from 1.0
        fn <- which((df[, predct_var_name] != df[, glb_rsp_var]) &
                    (df[, predct_var_name] == levels(df[, glb_rsp_var])[1]))
        df[fn, predct_erabs_var_name] <- abs(1 - df[fn, predct_prob_var_name])
        #rowIx <- which.max(df[fn, predct_erabs_var_name]); df[fn, c(glbFeatsId, glb_rsp_var, predct_var_name, predct_prob_var_name, predct_erabs_var_name)][rowIx, ]

        
        if (verbose) print(head(orderBy(reformulate(c("-", predct_erabs_var_name)), df)))
        
        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])
    }    
    
    if (glb_is_classification && !glb_is_binomial) {
        df[, predct_var_name] <- predict(mdl, newdata = df, type = "raw")
        probCls <- predict(mdl, newdata = df, type = "prob")        
        df[, predct_prob_var_name] <- NA
        for (cls in names(probCls)) {
            mask <- (df[, predct_var_name] == cls)
            df[mask, predct_prob_var_name] <- probCls[mask, cls]
        }    
        if (verbose) print(myplot_histogram(df, predct_prob_var_name, 
                                            fill_col_name = predct_var_name))
        if (verbose) print(myplot_histogram(df, predct_prob_var_name, 
                                            facet_frmla = paste0("~", glb_rsp_var)))
        
        df[, predct_error_var_name] <- df[, predct_var_name] != df[, glb_rsp_var]
        
        # if prediction is erroneous, measure predicted class prob from actual class prob
        df[, predct_erabs_var_name] <- 0
        for (cls in names(probCls)) {
            mask <- (df[, glb_rsp_var] == cls) & (df[, predct_error_var_name])
            df[mask, predct_erabs_var_name] <- probCls[mask, cls]
        }    

        df[, predct_accurate_var_name] <- (df[, glb_rsp_var] == df[, predct_var_name])        
    }

    return(df)
}    

#stop(here"); glb2Sav(); glbObsAll <- savObsAll; glbObsTrn <- savObsTrn; glbObsFit <- savObsFit; glbObsOOB <- savObsOOB; sav_models_df <- glb_models_df; glb_models_df <- sav_models_df; glb_featsimp_df <- sav_featsimp_df    

myget_category_stats <- function(obs_df, mdl_id, label) {
    require(dplyr)
    require(lazyeval)
    
    predct_var_name <- mygetPredictIds(glb_rsp_var, mdl_id)$value        
    predct_error_var_name <- mygetPredictIds(glb_rsp_var, mdl_id)$err.abs
    
    if (!predct_var_name %in% names(obs_df))
        obs_df <- glb_get_predictions(obs_df, mdl_id, glb_rsp_var)
    
    tmp_obs_df <- obs_df[, c(glbFeatsCategory, glb_rsp_var, 
                             predct_var_name, predct_error_var_name)]
#     tmp_obs_df <- obs_df %>%
#         dplyr::select_(glbFeatsCategory, glb_rsp_var, predct_var_name, predct_error_var_name) 
    #dplyr::rename(startprice.log10.predict.RFE.X.glmnet.err=error_abs_OOB)
    names(tmp_obs_df)[length(names(tmp_obs_df))] <- paste0("err.abs.", label)
    
    ret_ctgry_df <- tmp_obs_df %>%
        dplyr::group_by_(glbFeatsCategory) %>%
        dplyr::summarise_(#interp(~sum(abs(var)), var=as.name(glb_rsp_var)), 
            interp(~sum(var), var=as.name(paste0("err.abs.", label))), 
            interp(~mean(var), var=as.name(paste0("err.abs.", label))),
            interp(~n()))
    names(ret_ctgry_df) <- c(glbFeatsCategory, 
                             #paste0(glb_rsp_var, ".abs.", label, ".sum"),
                             paste0("err.abs.", label, ".sum"),                             
                             paste0("err.abs.", label, ".mean"), 
                             paste0(".n.", label))
    ret_ctgry_df <- dplyr::ungroup(ret_ctgry_df)
    #colSums(ret_ctgry_df[, -grep(glbFeatsCategory, names(ret_ctgry_df))])
    
    return(ret_ctgry_df)    
}
#print(colSums((ctgry_df <- myget_category_stats(obs_df=glbObsFit, mdl_id="", label="fit"))[, -grep(glbFeatsCategory, names(ctgry_df))]))

if (!is.null(glb_mdl_ensemble)) {
    fit.models_2_chunk_df <- myadd_chunk(fit.models_2_chunk_df, 
                            paste0("fit.models_2_", mdl_id_pfx), major.inc = TRUE, 
                                                label.minor = "ensemble")
    
    mdl_id_pfx <- "Ensemble"

    if (#(glb_is_regression) | 
        ((glb_is_classification) & (!glb_is_binomial)))
        stop("Ensemble models not implemented yet for multinomial classification")
    
    mygetEnsembleAutoMdlIds <- function() {
        tmp_models_df <- orderBy(get_model_sel_frmla(), glb_models_df)
        row.names(tmp_models_df) <- tmp_models_df$id
        mdl_threshold_pos <- 
            min(which(grepl("MFO|Random|Baseline", tmp_models_df$id))) - 1
        mdlIds <- tmp_models_df$id[1:mdl_threshold_pos]
        return(mdlIds[!grepl("Ensemble", mdlIds)])
    }
    
    if (glb_mdl_ensemble == "auto") {
        glb_mdl_ensemble <- mygetEnsembleAutoMdlIds()
        mdl_id_pfx <- paste0(mdl_id_pfx, ".auto")        
    } else if (grepl("^%<d-%", glb_mdl_ensemble)) {
        glb_mdl_ensemble <- eval(parse(text =
                        str_trim(unlist(strsplit(glb_mdl_ensemble, "%<d-%"))[2])))
    }
    
    for (mdl_id in glb_mdl_ensemble) {
        if (!(mdl_id %in% names(glb_models_lst))) {
            warning("Model ", mdl_id, " in glb_model_ensemble not found !")
            next
        }
        glbObsFit <- glb_get_predictions(df = glbObsFit, mdl_id, glb_rsp_var)
        glbObsOOB <- glb_get_predictions(df = glbObsOOB, mdl_id, glb_rsp_var)
    }
    
#mdl_id_pfx <- "Ensemble.RFE"; mdlId <- paste0(mdl_id_pfx, ".glmnet")
#glb_mdl_ensemble <- gsub(mygetPredictIds$value, "", grep("RFE\\.X\\.(?!Interact)", row.names(glb_featsimp_df), perl = TRUE, value = TRUE), fixed = TRUE)
#varImp(glb_models_lst[[mdlId]])
    
#cor_df <- data.frame(cor=cor(glbObsFit[, glb_rsp_var], glbObsFit[, paste(mygetPredictIds$value, glb_mdl_ensemble)], use="pairwise.complete.obs"))
#glbObsFit <- glb_get_predictions(df=glbObsFit, "Ensemble.glmnet", glb_rsp_var);print(colSums((ctgry_df <- myget_category_stats(obs_df=glbObsFit, mdl_id="Ensemble.glmnet", label="fit"))[, -grep(glbFeatsCategory, names(ctgry_df))]))
    
    ### bid0_sp
    #  Better than MFO; models.n=28; min.RMSE.fit=0.0521233; err.abs.fit.sum=7.3631895
    #  old: Top x from auto; models.n= 5; min.RMSE.fit=0.06311047; err.abs.fit.sum=9.5937080
    #  RFE only ;       models.n=16; min.RMSE.fit=0.05148588; err.abs.fit.sum=7.2875091
    #  RFE subset only ;models.n= 5; min.RMSE.fit=0.06040702; err.abs.fit.sum=9.059088
    #  RFE subset only ;models.n= 9; min.RMSE.fit=0.05933167; err.abs.fit.sum=8.7421288
    #  RFE subset only ;models.n=15; min.RMSE.fit=0.0584607; err.abs.fit.sum=8.5902066
    #  RFE subset only ;models.n=17; min.RMSE.fit=0.05496899; err.abs.fit.sum=8.0170431
    #  RFE subset only ;models.n=18; min.RMSE.fit=0.05441577; err.abs.fit.sum=7.837223
    #  RFE subset only ;models.n=16; min.RMSE.fit=0.05441577; err.abs.fit.sum=7.837223
    ### bid0_sp
    ### bid1_sp
    # "auto"; err.abs.fit.sum=76.699774; min.RMSE.fit=0.2186429
    # "RFE.X.*"; err.abs.fit.sum=; min.RMSE.fit=0.221114
    ### bid1_sp

    indep_vars <- paste(mygetPredictIds(glb_rsp_var)$value, glb_mdl_ensemble, sep = "")
    if (glb_is_classification)
        indep_vars <- paste(indep_vars, ".prob", sep = "")
    # Some models in glb_mdl_ensemble might not be fitted e.g. RFE.X.Interact
    indep_vars <- intersect(indep_vars, names(glbObsFit))
    
#     indep_vars <- grep(mygetPredictIds(glb_rsp_var)$value, names(glbObsFit), fixed=TRUE, value=TRUE)
#     if (glb_is_regression)
#         indep_vars <- indep_vars[!grepl("(err\\.abs|accurate)$", indep_vars)]
#     if (glb_is_classification && glb_is_binomial)
#         indep_vars <- grep("prob$", indep_vars, value=TRUE) else
#         indep_vars <- indep_vars[!grepl("err$", indep_vars)]

    #rfe_fit_ens_results <- myrun_rfe(glbObsFit, indep_vars)
    
    for (method in c("glm", "glmnet")) {
        for (trainControlMethod in 
             c("boot", "boot632", "cv", "repeatedcv"
               #, "LOOCV" # tuneLength * nrow(fitDF)
               , "LGOCV", "adaptive_cv"
               #, "adaptive_boot"  #error: adaptive$min should be less than 3 
               #, "adaptive_LGOCV" #error: adaptive$min should be less than 3 
               )) {
            #sav_models_df <- glb_models_df; all.equal(sav_models_df, glb_models_df)
            #glb_models_df <- sav_models_df; print(glb_models_df$id)
                
            if ((method == "glm") && (trainControlMethod != "repeatedcv"))
                # glm used only to identify outliers
                next
            
            ret_lst <- myfit_mdl(
                mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                    id.prefix = paste0(mdl_id_pfx, ".", trainControlMethod), 
                    type = glb_model_type, tune.df = NULL,
                    trainControl.method = trainControlMethod,
                    trainControl.number = glb_rcv_n_folds,
                    trainControl.repeats = glb_rcv_n_repeats,
                    trainControl.classProbs = glb_is_classification,
                    trainControl.summaryFunction = glbMdlMetricSummaryFn,
                    train.metric = glbMdlMetricSummary, 
                    train.maximize = glbMdlMetricMaximize,    
                    train.method = method)),
                indep_vars = indep_vars, rsp_var = glb_rsp_var, 
                fit_df = glbObsFit, OOB_df = glbObsOOB)
        }
    }
    dsp_models_df <- get_dsp_models_df()
}

if (is.null(glb_sel_mdl_id)) 
    glb_sel_mdl_id <- dsp_models_df[1, "id"] else 
    print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
```

```
## [1] "User specified selection: All.X##rcv#glmnet"
```

```r
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

![](Houses_tmplt_files/figure-html/fit.models_2-2.png) 

```
##             Length Class      Mode     
## a0           60    -none-     numeric  
## beta        120    dgCMatrix  S4       
## df           60    -none-     numeric  
## dim           2    -none-     numeric  
## lambda       60    -none-     numeric  
## dev.ratio    60    -none-     numeric  
## nulldev       1    -none-     numeric  
## npasses       1    -none-     numeric  
## jerr          1    -none-     numeric  
## offset        1    -none-     logical  
## call          5    -none-     call     
## nobs          1    -none-     numeric  
## lambdaOpt     1    -none-     numeric  
## xNames        2    -none-     character
## problemType   1    -none-     character
## tuneValue     2    data.frame list     
## obsLevels     1    -none-     logical  
## [1] "min lambda > lambdaOpt:"
##   (Intercept)          .pos        .rnorm 
## 526632.911885      2.059212  -2697.858937 
## [1] "max lambda < lambdaOpt:"
## [1] "Feats mismatch between coefs_left & rght:"
## [1] "(Intercept)" ".pos"        ".rnorm"
```

```
## [1] TRUE
```

```r
# From here to save(), this should all be in one function
#   these are executed in the same seq twice more:
#       fit.data.training & predict.data.new chunks
print(sprintf("%s fit prediction diagnostics:", glb_sel_mdl_id))
```

```
## [1] "All.X##rcv#glmnet fit prediction diagnostics:"
```

```r
glbObsFit <- glb_get_predictions(df = glbObsFit, mdl_id = glb_sel_mdl_id, 
                                 rsp_var = glb_rsp_var)
print(sprintf("%s OOB prediction diagnostics:", glb_sel_mdl_id))
```

```
## [1] "All.X##rcv#glmnet OOB prediction diagnostics:"
```

```r
glbObsOOB <- glb_get_predictions(df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
                                     rsp_var = glb_rsp_var)

print(glb_featsimp_df <- myget_feats_importance(mdl = glb_sel_mdl, featsimp_df = NULL))
```

```
##        All.X..rcv.glmnet.imp imp All.X##rcv#glmnet.imp
## .pos                     100 100                   100
## .rnorm                     0   0                     0
```

```r
#mdl_id <-"RFE.X.glmnet"; glb_featsimp_df <- myget_feats_importance(glb_models_lst[[mdl_id]], glb_featsimp_df); glb_featsimp_df[, paste0(mdl_id, ".imp")] <- glb_featsimp_df$imp; print(glb_featsimp_df)
#print(head(sbst_featsimp_df <- subset(glb_featsimp_df, is.na(RFE.X.glmnet.imp) | (abs(RFE.X.YeoJohnson.glmnet.imp - RFE.X.glmnet.imp) > 0.0001), select=-imp)))
#print(orderBy(~ -cor.y.abs, subset(glb_feats_df, id %in% c(row.names(sbst_featsimp_df), "startprice.dcm1.is9", "D.weight.post.stop.sum"))))

# Used again in fit.data.training & predict.data.new chunks
glb_analytics_diag_plots <- function(obs_df, mdl_id, prob_threshold=NULL) {
    if (!is.null(featsimp_df <- glb_featsimp_df)) {
        featsimp_df$feat <- gsub("`(.*?)`", "\\1", row.names(featsimp_df))    
        featsimp_df$feat.interact <- gsub("(.*?):(.*)", "\\2", featsimp_df$feat)
        featsimp_df$feat <- gsub("(.*?):(.*)", "\\1", featsimp_df$feat)    
        featsimp_df$feat.interact <- 
            ifelse(featsimp_df$feat.interact == featsimp_df$feat, 
                                            NA, featsimp_df$feat.interact)
        featsimp_df$feat <- 
            gsub("(.*?)\\.fctr(.*)", "\\1\\.fctr", featsimp_df$feat)
        featsimp_df$feat.interact <- 
            gsub("(.*?)\\.fctr(.*)", "\\1\\.fctr", featsimp_df$feat.interact) 
        featsimp_df <- orderBy(~ -imp.max, 
            summaryBy(imp ~ feat + feat.interact, data=featsimp_df,
                      FUN=max))    
        #rex_str=":(.*)"; txt_vctr=tail(featsimp_df$feat); ret_lst <- regexec(rex_str, txt_vctr); ret_lst <- regmatches(txt_vctr, ret_lst); ret_vctr <- sapply(1:length(ret_lst), function(pos_ix) ifelse(length(ret_lst[[pos_ix]]) > 0, ret_lst[[pos_ix]], "")); print(ret_vctr <- ret_vctr[ret_vctr != ""])    
        
        featsimp_df <- subset(featsimp_df, !is.na(imp.max))
        if (nrow(featsimp_df) > 5) {
            warning("Limiting important feature scatter plots to 5 out of ",
                    nrow(featsimp_df))
            featsimp_df <- head(featsimp_df, 5)
        }
        
    #     if (!all(is.na(featsimp_df$feat.interact)))
    #         stop("not implemented yet")
        rsp_var_out <- mygetPredictIds(glb_rsp_var, mdl_id)$value
        for (var in featsimp_df$feat) {
            plot_df <- melt(obs_df, id.vars = var, 
                            measure.vars = c(glb_rsp_var, rsp_var_out))
    
            print(myplot_scatter(plot_df, var, "value", colorcol_name = "variable",
                                facet_colcol_name = "variable", jitter = TRUE) + 
                          guides(color = FALSE))
        }
    }
    
    if (glb_is_regression) {
        if (is.null(featsimp_df) || (nrow(featsimp_df) == 0))
            warning("No important features in glb_fin_mdl") else
            print(myplot_prediction_regression(df=obs_df, 
                        feat_x=ifelse(nrow(featsimp_df) > 1, featsimp_df$feat[2],
                                      ".rownames"), 
                                               feat_y=featsimp_df$feat[1],
                        rsp_var=glb_rsp_var, rsp_var_out=rsp_var_out,
                        id_vars=glbFeatsId)
    #               + facet_wrap(reformulate(featsimp_df$feat[2])) # if [1 or 2] is a factor
    #               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
                  )
    }    
    
    if (glb_is_classification) {
        if (is.null(featsimp_df) || (nrow(featsimp_df) == 0))
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df = obs_df, 
                                feat_x = ifelse(nrow(featsimp_df) > 1, 
                                                featsimp_df$feat[2], ".rownames"),
                                               feat_y = featsimp_df$feat[1],
                                                rsp_var = glb_rsp_var, 
                                                rsp_var_out = rsp_var_out, 
                                                id_vars = glbFeatsId,
                                                prob_threshold = prob_threshold))
    }    
}

if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id, 
            prob_threshold = glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                           "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id)                  
```

![](Houses_tmplt_files/figure-html/fit.models_2-3.png) ![](Houses_tmplt_files/figure-html/fit.models_2-4.png) 

```
##               id            date   price bedrooms bathrooms sqft_living
## 18200 3625059152 20141230T000000 3300000        3      3.25        4220
## 7192  7159200005 20140507T000000 3200000        7      4.50        6210
## 16970 3025059093 20140729T000000 3100000        5      5.25        5090
## 11976 1118000340 20150408T000000 3000000        5      3.75        4590
## 2975  3625049014 20140829T000000 2950000        4      3.50        4860
##       sqft_lot floors waterfront view condition grade sqft_above
## 18200    41300    1.0          1    4         4    11       2460
## 7192      8856    2.5          0    2         5    11       4760
## 16970    23669    2.0          0    0         3    12       5090
## 11976    11265    2.0          0    0         4    11       3450
## 2975     23885    2.0          0    0         3    12       4860
##       sqft_basement yr_built yr_renovated zipcode     lat     long
## 18200          1760     1958         1987   98008 47.6083 -122.110
## 7192           1450     1910            0   98109 47.6307 -122.354
## 16970             0     2006            0   98004 47.6297 -122.216
## 11976          1140     1927            0   98112 47.6389 -122.288
## 2975              0     1996            0   98039 47.6172 -122.230
##       sqft_living15 sqft_lot15 .src     .rnorm  .pos
## 18200          3810      30401 Test  1.5699436 21062
## 7192           2940       5400 Test  0.2160286 19106
## 16970          3830      22605 Test -1.3551279 20831
## 11976          3870       8996 Test -0.4948976 19932
## 2975           3580      16054 Test  0.2290843 18380
##                          id.date .category price.All.X..rcv.glmnet
## 18200 3625059152#20141230T000000    .dummy                565768.6
## 7192  7159200005#20140507T000000    .dummy                565393.4
## 16970 3025059093#20140729T000000    .dummy                573184.3
## 11976 1118000340#20150408T000000    .dummy                569012.3
## 2975  3625049014#20140829T000000    .dummy                563863.2
##       price.All.X..rcv.glmnet.err price.All.X..rcv.glmnet.err.abs
## 18200                     2734231                         2734231
## 7192                      2634607                         2634607
## 16970                     2526816                         2526816
## 11976                     2430988                         2430988
## 2975                      2386137                         2386137
##       price.All.X..rcv.glmnet.is.acc                     .label
## 18200                          FALSE 3625059152#20141230T000000
## 7192                           FALSE 7159200005#20140507T000000
## 16970                          FALSE 3025059093#20140729T000000
## 11976                          FALSE 1118000340#20150408T000000
## 2975                           FALSE 3625049014#20140829T000000
```

![](Houses_tmplt_files/figure-html/fit.models_2-5.png) 

```r
if (!is.null(glbFeatsCategory)) {
    glbLvlCategory <- merge(glbLvlCategory, 
            myget_category_stats(obs_df = glbObsFit, mdl_id = glb_sel_mdl_id, 
                                 label = "fit"), 
                            by = glbFeatsCategory, all = TRUE)
    row.names(glbLvlCategory) <- glbLvlCategory[, glbFeatsCategory]
    glbLvlCategory <- merge(glbLvlCategory, 
            myget_category_stats(obs_df = glbObsOOB, mdl_id = glb_sel_mdl_id,
                                 label="OOB"),
                          #by=glbFeatsCategory, all=TRUE) glb_ctgry-df already contains .n.OOB ?
                          all = TRUE)
    row.names(glbLvlCategory) <- glbLvlCategory[, glbFeatsCategory]
    if (any(grepl("OOB", glbMdlMetricsEval)))
        print(orderBy(~-err.abs.OOB.mean, glbLvlCategory)) else
            print(orderBy(~-err.abs.fit.mean, glbLvlCategory))
    print(colSums(glbLvlCategory[, -grep(glbFeatsCategory, names(glbLvlCategory))]))
}
```

```
##        .category .n.OOB .n.Fit .n.Tst .freqRatio.Fit .freqRatio.OOB
## .dummy    .dummy   3772  17841   3772              1              1
##        .freqRatio.Tst err.abs.fit.sum err.abs.fit.mean .n.fit
## .dummy              1      4266379442         239133.4  17841
##        err.abs.OOB.sum err.abs.OOB.mean
## .dummy       845864513         224248.3
##           .n.OOB           .n.Fit           .n.Tst   .freqRatio.Fit 
##           3772.0          17841.0           3772.0              1.0 
##   .freqRatio.OOB   .freqRatio.Tst  err.abs.fit.sum err.abs.fit.mean 
##              1.0              1.0     4266379442.3         239133.4 
##           .n.fit  err.abs.OOB.sum err.abs.OOB.mean 
##          17841.0      845864513.4         224248.3
```

```r
write.csv(glbObsOOB[, c(glbFeatsId, 
                grep(glb_rsp_var, names(glbObsOOB), fixed=TRUE, value=TRUE))], 
    paste0(gsub(".", "_", paste0(glb_out_pfx, glb_sel_mdl_id), fixed=TRUE), 
           "_OOBobs.csv"), row.names=FALSE)

fit.models_2_chunk_df <- 
    myadd_chunk(NULL, "fit.models_2_bgn", label.minor = "teardown")
```

```
##              label step_major step_minor label_minor    bgn end elapsed
## 1 fit.models_2_bgn          1          0    teardown 72.737  NA      NA
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.models", major.inc=FALSE)
```

```
##         label step_major step_minor label_minor    bgn    end elapsed
## 17 fit.models          8          2           2 61.709 72.748  11.039
## 18 fit.models          8          3           3 72.748     NA      NA
```


```r
# if (sum(is.na(glbObsAll$D.P.http)) > 0)
#         stop("fit.models_3: Why is this happening ?")

#stop(here"); glb2Sav()
sync_glb_obs_df <- function() {
    # Merge or cbind ?
    for (col in setdiff(names(glbObsFit), names(glbObsTrn)))
        glbObsTrn[glbObsTrn$.lcn == "Fit", col] <<- glbObsFit[, col]
    for (col in setdiff(names(glbObsFit), names(glbObsAll)))
        glbObsAll[glbObsAll$.lcn == "Fit", col] <<- glbObsFit[, col]
    if (all(is.na(glbObsNew[, glb_rsp_var])))
        for (col in setdiff(names(glbObsOOB), names(glbObsTrn)))
            glbObsTrn[glbObsTrn$.lcn == "OOB", col] <<- glbObsOOB[, col]
    for (col in setdiff(names(glbObsOOB), names(glbObsAll)))
        glbObsAll[glbObsAll$.lcn == "OOB", col] <<- glbObsOOB[, col]
}
sync_glb_obs_df()
    
print(setdiff(names(glbObsNew), names(glbObsAll)))
```

```
## character(0)
```

```r
if (glb_save_envir)
    save(glb_feats_df, 
         glbObsAll, #glbObsTrn, glbObsFit, glbObsOOB, glbObsNew,
         glb_models_df, dsp_models_df, glb_models_lst, glb_sel_mdl, glb_sel_mdl_id,
         glb_model_type,
        file=paste0(glb_out_pfx, "selmdl_dsk.RData"))
#load(paste0(glb_out_pfx, "selmdl_dsk.RData"))

rm(ret_lst)
```

```
## Warning in rm(ret_lst): object 'ret_lst' not found
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](Houses_tmplt_files/figure-html/fit.models_3-1.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=TRUE)
```

```
##                label step_major step_minor label_minor    bgn    end
## 18        fit.models          8          3           3 72.748 78.717
## 19 fit.data.training          9          0           0 78.717     NA
##    elapsed
## 18   5.969
## 19      NA
```

## Step `9.0: fit data training`

```r
#load(paste0(glb_inp_pfx, "dsk.RData"))

if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_models_lst[[glb_fin_mdl_id]]
} else 
# if (nrow(glbObsFit) + length(glbObsFitOutliers) == nrow(glbObsTrn))
if (!all(is.na(glbObsNew[, glb_rsp_var])))
{    
    warning("Final model same as glb_sel_mdl_id")
    glb_fin_mdl_id <- paste0("Final.", glb_sel_mdl_id)
    glb_fin_mdl <- glb_sel_mdl
    glb_models_lst[[glb_fin_mdl_id]] <- glb_fin_mdl
} else {    
            if (grepl("RFE\\.X", names(glbMdlFamilies))) {
                indep_vars <- myadjust_interaction_feats(subset(glb_feats_df, 
                                                    !nzv & (exclude.as.feat != 1))[, "id"])
                rfe_trn_results <- 
                    myrun_rfe(glbObsTrn, indep_vars, glbRFESizes[["Final"]])
                if (!isTRUE(all.equal(sort(predictors(rfe_trn_results)),
                                      sort(predictors(rfe_fit_results))))) {
                    print("Diffs predictors(rfe_trn_results) vs. predictors(rfe_fit_results):")
                    print(setdiff(predictors(rfe_trn_results), predictors(rfe_fit_results)))
                    print("Diffs predictors(rfe_fit_results) vs. predictors(rfe_trn_results):")
                    print(setdiff(predictors(rfe_fit_results), predictors(rfe_trn_results)))
            }
        }
    # }    

    if (grepl("Ensemble", glb_sel_mdl_id)) {
        # Find which models are relevant
        mdlimp_df <- subset(myget_feats_importance(glb_sel_mdl), imp > 5)
        # Fit selected models on glbObsTrn
        for (mdl_id in gsub(".prob", "", 
gsub(mygetPredictIds(glb_rsp_var)$value, "", row.names(mdlimp_df), fixed = TRUE),
                            fixed = TRUE)) {
            mdl_id_components <- unlist(strsplit(mdl_id, "[.]"))
            mdlIdPfx <- paste0(c(head(mdl_id_components, -1), "Train"), 
                               collapse = ".")
            if (grepl("RFE\\.X\\.", mdlIdPfx)) 
                mdlIndepVars <- myadjust_interaction_feats(myextract_actual_feats(
                    predictors(rfe_trn_results))) else
                mdlIndepVars <- trim(unlist(
            strsplit(glb_models_df[glb_models_df$id == mdl_id, "feats"], "[,]")))
            ret_lst <- 
                myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                        id.prefix = mdlIdPfx, 
                        type = glb_model_type, tune.df = glbMdlTuneParams,
                        trainControl.method = "repeatedcv",
                        trainControl.number = glb_rcv_n_folds,
                        trainControl.repeats = glb_rcv_n_repeats,
                        trainControl.classProbs = glb_is_classification,
                        trainControl.summaryFunction = glbMdlMetricSummaryFn,
                        train.metric = glbMdlMetricSummary, 
                        train.maximize = glbMdlMetricMaximize,    
                        train.method = tail(mdl_id_components, 1))),
                    indep_vars = mdlIndepVars,
                    rsp_var = glb_rsp_var, 
                    fit_df = glbObsTrn, OOB_df = NULL)
            
            glbObsTrn <- glb_get_predictions(df = glbObsTrn,
                                                mdl_id = tail(glb_models_df$id, 1), 
                                                rsp_var = glb_rsp_var,
                                                prob_threshold_def = 
                    subset(glb_models_df, id == mdl_id)$opt.prob.threshold.OOB)
            glbObsNew <- glb_get_predictions(df = glbObsNew,
                                                mdl_id = tail(glb_models_df$id, 1), 
                                                rsp_var = glb_rsp_var,
                                                prob_threshold_def = 
                    subset(glb_models_df, id == mdl_id)$opt.prob.threshold.OOB)
        }    
    }
    
    # "Final" model
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the mdl_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
        
    if (grepl("Ensemble", glb_sel_mdl_id)) {
        # Find which models are relevant
        mdlimp_df <- subset(myget_feats_importance(glb_sel_mdl), imp > 5)
        if (glb_is_classification && glb_is_binomial)
            indep_vars_vctr <- gsub("(.*)\\.(.*)\\.prob", "\\1\\.Train\\.\\2\\.prob",
                                    row.names(mdlimp_df)) else
            indep_vars_vctr <- gsub("(.*)\\.(.*)", "\\1\\.Train\\.\\2",
                                    row.names(mdlimp_df))
    } else 
    if (grepl("RFE.X", glb_sel_mdl_id, fixed = TRUE)) {
        indep_vars_vctr <- myextract_actual_feats(predictors(rfe_trn_results))
    } else indep_vars_vctr <- 
                trim(unlist(strsplit(glb_models_df[glb_models_df$id ==
                                                   glb_sel_mdl_id
                                                   , "feats"], "[,]")))
        
    if (!is.null(glb_preproc_methods) &&
        ((match_pos <- regexpr(gsub(".", "\\.", 
                                    paste(glb_preproc_methods, collapse = "|"),
                                   fixed = TRUE), glb_sel_mdl_id)) != -1))
        ths_preProcess <- str_sub(glb_sel_mdl_id, match_pos, 
                                match_pos + attr(match_pos, "match.length") - 1) else
        ths_preProcess <- NULL                                      

    mdl_id_pfx <- ifelse(grepl("Ensemble", glb_sel_mdl_id),
                                   "Final.Ensemble", "Final")
    trnobs_df <- if (is.null(glbObsTrnOutliers[[mdl_id_pfx]])) glbObsTrn else 
        glbObsTrn[!(glbObsTrn[, glbFeatsId] %in%
                            glbObsTrnOutliers[[mdl_id_pfx]]), ]
        
    # Force fitting of Final.glm to identify outliers
    method_vctr <- unique(c(myparseMdlId(glb_sel_mdl_id)$alg, glbMdlFamilies[["Final"]]))
    for (method in method_vctr) {
        #source("caret_nominalTrainWorkflow.R")
        
        # glmnet requires at least 2 indep vars
        if ((length(indep_vars_vctr) == 1) && (method %in% "glmnet"))
            next
        
        ret_lst <- 
            myfit_mdl(mdl_specs_lst = myinit_mdl_specs_lst(mdl_specs_lst = list(
                    id.prefix = mdl_id_pfx, 
                    type = glb_model_type, trainControl.method = "repeatedcv",
                    trainControl.number = glb_rcv_n_folds, 
                    trainControl.repeats = glb_rcv_n_repeats,
                    trainControl.classProbs = glb_is_classification,
                    trainControl.summaryFunction = glbMdlMetricSummaryFn,
                    trainControl.allowParallel = glbMdlAllowParallel,
                    train.metric = glbMdlMetricSummary, 
                    train.maximize = glbMdlMetricMaximize,    
                    train.method = method,
                    train.preProcess = ths_preProcess)),
                indep_vars = indep_vars_vctr, rsp_var = glb_rsp_var, 
                fit_df = trnobs_df, OOB_df = NULL)
    }
        
    if ((length(method_vctr) == 1) || (method != "glm")) {
        glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]] 
        glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "id"]
    }
}
```

```
## Warning: Final model same as glb_sel_mdl_id
```

```r
rm(ret_lst)
```

```
## Warning in rm(ret_lst): object 'ret_lst' not found
```

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "fit.data.training", major.inc=FALSE)
```

```
##                label step_major step_minor label_minor    bgn    end
## 19 fit.data.training          9          0           0 78.717 79.142
## 20 fit.data.training          9          1           1 79.143     NA
##    elapsed
## 19   0.426
## 20      NA
```


```r
#stop(here"); glb2Sav()
if (glb_is_classification && glb_is_binomial) 
    prob_threshold <- glb_models_df[glb_models_df$id == glb_sel_mdl_id,
                                        "opt.prob.threshold.OOB"] else 
    prob_threshold <- NULL

if (grepl("Ensemble", glb_fin_mdl_id)) {
    # Get predictions for each model in ensemble; Outliers that have been moved to OOB might not have been predicted yet
    mdlEnsembleComps <- unlist(str_split(subset(glb_models_df, 
                                                id == glb_fin_mdl_id)$feats, ","))
    if (glb_is_classification && glb_is_binomial)
        mdlEnsembleComps <- gsub("\\.prob$", "", mdlEnsembleComps)
    mdlEnsembleComps <- gsub(paste0("^", 
                        gsub(".", "\\.", mygetPredictIds(glb_rsp_var)$value, fixed = TRUE)),
                             "", mdlEnsembleComps)
    for (mdl_id in mdlEnsembleComps) {
        glbObsTrn <- glb_get_predictions(df = glbObsTrn, mdl_id = mdl_id, 
                                            rsp_var = glb_rsp_var,
                                            prob_threshold_def = prob_threshold)
        glbObsNew <- glb_get_predictions(df = glbObsNew, mdl_id = mdl_id, 
                                            rsp_var = glb_rsp_var,
                                            prob_threshold_def = prob_threshold)
    }    
}
glbObsTrn <- glb_get_predictions(df = glbObsTrn, mdl_id = glb_fin_mdl_id, 
                                     rsp_var = glb_rsp_var,
                                    prob_threshold_def = prob_threshold)

glb_featsimp_df <- myget_feats_importance(mdl=glb_fin_mdl,
                                          featsimp_df=glb_featsimp_df)
glb_featsimp_df[, paste0(glb_fin_mdl_id, ".imp")] <- glb_featsimp_df$imp
print(glb_featsimp_df)
```

```
##        All.X..rcv.glmnet.imp.x All.X##rcv#glmnet.imp
## .pos                       100                   100
## .rnorm                       0                     0
##        All.X..rcv.glmnet.imp.y imp Final.All.X##rcv#glmnet.imp
## .pos                       100 100                         100
## .rnorm                       0   0                           0
```

```r
if (glb_is_classification && glb_is_binomial)
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id, 
            prob_threshold=glb_models_df[glb_models_df$id == glb_sel_mdl_id, 
                                         "opt.prob.threshold.OOB"]) else
    glb_analytics_diag_plots(obs_df=glbObsTrn, mdl_id=glb_fin_mdl_id)                  
```

![](Houses_tmplt_files/figure-html/fit.data.training_1-1.png) ![](Houses_tmplt_files/figure-html/fit.data.training_1-2.png) 

```
##              id            date   price bedrooms bathrooms sqft_living
## 7253 6762700020 20141013T000000 7700000        6      8.00       12050
## 3915 9808700762 20140611T000000 7062500        5      4.50       10040
## 9255 9208900037 20140919T000000 6885000        6      7.75        9890
## 4412 2470100110 20140804T000000 5570000        5      5.75        9200
## 1449 8907500070 20150413T000000 5350000        5      5.00        8000
##      sqft_lot floors condition grade sqft_above yr_built zipcode     lat
## 7253    27600    2.5         4    13       8570     1910   98102 47.6298
## 3915    37325    2.0         3    11       7680     1940   98004 47.6500
## 9255    31374    2.0         3    13       8860     2001   98039 47.6305
## 4412    35069    2.0         3    13       6200     2001   98039 47.6289
## 1449    23985    2.0         3    12       6720     2009   98004 47.6232
##          long sqft_living15 sqft_lot15  .src    .rnorm .pos
## 7253 -122.323          3940       8800 Train 0.4283131 5972
## 3915 -122.214          3930      25449 Train 0.7554864 3215
## 9255 -122.240          4540      42730 Train 0.6897146 7619
## 4412 -122.233          3560      24345 Train 0.3178016 3626
## 1449 -122.220          4600      21750 Train 0.1298099 1180
##                         id.date .lcn .category waterfront view
## 7253 6762700020#20141013T000000  Fit    .dummy          0    3
## 3915 9808700762#20140611T000000  Fit    .dummy          1    2
## 9255 9208900037#20140919T000000  Fit    .dummy          0    4
## 4412 2470100110#20140804T000000  Fit    .dummy          0    0
## 1449 8907500070#20150413T000000  Fit    .dummy          0    4
##      sqft_basement yr_renovated price.All.X..rcv.glmnet
## 7253          3480         1987                537775.0
## 3915          2360         2001                531215.1
## 9255          1030            0                540461.3
## 4412          3000            0                533242.2
## 1449          1280            0                528712.6
##      price.All.X..rcv.glmnet.err price.All.X..rcv.glmnet.err.abs
## 7253                    -7162225                         7162225
## 3915                    -6531285                         6531285
## 9255                    -6344539                         6344539
## 4412                    -5036758                         5036758
## 1449                    -4821287                         4821287
##      price.All.X..rcv.glmnet.is.acc price.Final.All.X..rcv.glmnet
## 7253                          FALSE                      537775.0
## 3915                          FALSE                      531215.1
## 9255                          FALSE                      540461.3
## 4412                          FALSE                      533242.2
## 1449                          FALSE                      528712.6
##      price.Final.All.X..rcv.glmnet.err
## 7253                           7162225
## 3915                           6531285
## 9255                           6344539
## 4412                           5036758
## 1449                           4821287
##      price.Final.All.X..rcv.glmnet.err.abs
## 7253                               7162225
## 3915                               6531285
## 9255                               6344539
## 4412                               5036758
## 1449                               4821287
##      price.Final.All.X..rcv.glmnet.is.acc                     .label
## 7253                                FALSE 6762700020#20141013T000000
## 3915                                FALSE 9808700762#20140611T000000
## 9255                                FALSE 9208900037#20140919T000000
## 4412                                FALSE 2470100110#20140804T000000
## 1449                                FALSE 8907500070#20150413T000000
```

![](Houses_tmplt_files/figure-html/fit.data.training_1-3.png) 

```r
dsp_feats_vctr <- c(NULL)
for(var in grep(".imp", names(glb_feats_df), fixed=TRUE, value=TRUE))
    dsp_feats_vctr <- union(dsp_feats_vctr, 
                            glb_feats_df[!is.na(glb_feats_df[, var]), "id"])

# print(glbObsTrn[glbObsTrn$UniqueID %in% FN_OOB_ids, 
#                     grep(glb_rsp_var, names(glbObsTrn), value=TRUE)])

print(setdiff(names(glbObsTrn), names(glbObsAll)))
```

```
## [1] "price.Final.All.X..rcv.glmnet"        
## [2] "price.Final.All.X..rcv.glmnet.err"    
## [3] "price.Final.All.X..rcv.glmnet.err.abs"
## [4] "price.Final.All.X..rcv.glmnet.is.acc"
```

```r
for (col in setdiff(names(glbObsTrn), names(glbObsAll)))
    # Merge or cbind ?
    glbObsAll[glbObsAll$.src == "Train", col] <- glbObsTrn[, col]

print(setdiff(names(glbObsFit), names(glbObsAll)))
```

```
## character(0)
```

```r
print(setdiff(names(glbObsOOB), names(glbObsAll)))
```

```
## character(0)
```

```r
for (col in setdiff(names(glbObsOOB), names(glbObsAll)))
    # Merge or cbind ?
    glbObsAll[glbObsAll$.lcn == "OOB", col] <- glbObsOOB[, col]
    
print(setdiff(names(glbObsNew), names(glbObsAll)))
```

```
## character(0)
```

```r
if (glb_save_envir)
    save(glb_feats_df, glbObsAll, 
         #glbObsTrn, glbObsFit, glbObsOOB, glbObsNew,
         glb_models_df, dsp_models_df, glb_models_lst, glb_model_type,
         glb_sel_mdl, glb_sel_mdl_id,
         glb_fin_mdl, glb_fin_mdl_id,
        file = paste0(glb_out_pfx, "dsk.RData"))

#glb2Sav(); all.equal(savObsAll, glbObsAll); all.equal(sav_models_lst, glb_models_lst)
#load(file = paste0(glb_out_pfx, "dsk_knitr.RData"))
#cmpCols <- names(glbObsAll)[!grepl("\\.Final\\.", names(glbObsAll))]; all.equal(savObsAll[, cmpCols], glbObsAll[, cmpCols]); all.equal(savObsAll[, "H.P.http"], glbObsAll[, "H.P.http"]); 

replay.petrisim(pn = glb_analytics_pn, 
    replay.trans = (glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord = TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](Houses_tmplt_files/figure-html/fit.data.training_1-4.png) 

```r
glb_chunks_df <- myadd_chunk(glb_chunks_df, "predict.data.new", major.inc = TRUE)
```

```
##                label step_major step_minor label_minor    bgn    end
## 20 fit.data.training          9          1           1 79.143 99.325
## 21  predict.data.new         10          0           0 99.325     NA
##    elapsed
## 20  20.182
## 21      NA
```

## Step `10.0: predict data new`
![](Houses_tmplt_files/figure-html/predict.data.new-1.png) ![](Houses_tmplt_files/figure-html/predict.data.new-2.png) 

```
##               id            date   price bedrooms bathrooms sqft_living
## 18200 3625059152 20141230T000000 3300000        3      3.25        4220
## 7192  7159200005 20140507T000000 3200000        7      4.50        6210
## 16970 3025059093 20140729T000000 3100000        5      5.25        5090
## 11976 1118000340 20150408T000000 3000000        5      3.75        4590
## 2975  3625049014 20140829T000000 2950000        4      3.50        4860
##       sqft_lot floors condition grade sqft_above yr_built zipcode     lat
## 18200    41300    1.0         4    11       2460     1958   98008 47.6083
## 7192      8856    2.5         5    11       4760     1910   98109 47.6307
## 16970    23669    2.0         3    12       5090     2006   98004 47.6297
## 11976    11265    2.0         4    11       3450     1927   98112 47.6389
## 2975     23885    2.0         3    12       4860     1996   98039 47.6172
##           long sqft_living15 sqft_lot15 .src     .rnorm  .pos
## 18200 -122.110          3810      30401 Test  1.5699436 21062
## 7192  -122.354          2940       5400 Test  0.2160286 19106
## 16970 -122.216          3830      22605 Test -1.3551279 20831
## 11976 -122.288          3870       8996 Test -0.4948976 19932
## 2975  -122.230          3580      16054 Test  0.2290843 18380
##                          id.date .lcn .category
## 18200 3625059152#20141230T000000  OOB    .dummy
## 7192  7159200005#20140507T000000  OOB    .dummy
## 16970 3025059093#20140729T000000  OOB    .dummy
## 11976 1118000340#20150408T000000  OOB    .dummy
## 2975  3625049014#20140829T000000  OOB    .dummy
##       price.Final.All.X..rcv.glmnet price.Final.All.X..rcv.glmnet.err
## 18200                      565768.6                           2734231
## 7192                       565393.4                           2634607
## 16970                      573184.3                           2526816
## 11976                      569012.3                           2430988
## 2975                       563863.2                           2386137
##       price.Final.All.X..rcv.glmnet.err.abs
## 18200                               2734231
## 7192                                2634607
## 16970                               2526816
## 11976                               2430988
## 2975                                2386137
##       price.Final.All.X..rcv.glmnet.is.acc                     .label
## 18200                                FALSE 3625059152#20141230T000000
## 7192                                 FALSE 7159200005#20140507T000000
## 16970                                FALSE 3025059093#20140729T000000
## 11976                                FALSE 1118000340#20150408T000000
## 2975                                 FALSE 3625049014#20140829T000000
```

![](Houses_tmplt_files/figure-html/predict.data.new-3.png) 

```
## Loading required package: stringr
```

```
## [1] "glb_sel_mdl_id: All.X##rcv#glmnet"
```

```
## [1] "glb_fin_mdl_id: Final.All.X##rcv#glmnet"
```

```
## [1] "Cross Validation issues:"
## Max.cor.Y.rcv.1X1###glmnet 
##                          0
```

```
##                            min.RMSE.OOB max.R.sq.OOB max.Adj.R.sq.fit
## Max.cor.Y##rcv#rpart           308953.6   0.00000000               NA
## Low.cor.X##rcv#glmnet          311717.0  -0.01796882     0.0007397307
## All.X##rcv#glmnet              311717.0  -0.01796882     0.0007397307
## Max.cor.Y.rcv.1X1###glmnet     311717.0  -0.01796882     0.0007397307
## All.X##rcv#glm                 311735.3  -0.01808814     0.0007397671
##                            min.RMSE.fit
## Max.cor.Y##rcv#rpart           378063.0
## Low.cor.X##rcv#glmnet          378055.2
## All.X##rcv#glmnet              378055.2
## Max.cor.Y.rcv.1X1###glmnet     378150.3
## All.X##rcv#glm                 378055.4
```

```
## [1] "All.X##rcv#glmnet OOB RMSE: 311717.0444"
```

```
##        .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.OOB .n.Tst
## .dummy              1              1              1  17841   3772   3772
##        .n.fit .n.new .n.trn err.abs.OOB.mean err.abs.fit.mean
## .dummy  17841   3772  17841         224248.3         239133.4
##        err.abs.new.mean err.abs.trn.mean err.abs.OOB.sum err.abs.fit.sum
## .dummy         224248.3         239133.4       845864513      4266379442
##        err.abs.new.sum err.abs.trn.sum
## .dummy       845864513      4266379442
##   .freqRatio.Fit   .freqRatio.OOB   .freqRatio.Tst           .n.Fit 
##              1.0              1.0              1.0          17841.0 
##           .n.OOB           .n.Tst           .n.fit           .n.new 
##           3772.0           3772.0          17841.0           3772.0 
##           .n.trn err.abs.OOB.mean err.abs.fit.mean err.abs.new.mean 
##          17841.0         224248.3         239133.4         224248.3 
## err.abs.trn.mean  err.abs.OOB.sum  err.abs.fit.sum  err.abs.new.sum 
##         239133.4      845864513.4     4266379442.3      845864513.4 
##  err.abs.trn.sum 
##     4266379442.3 
##        .freqRatio.Fit .freqRatio.OOB .freqRatio.Tst .n.Fit .n.OOB .n.Tst
## .dummy              1              1              1  17841   3772   3772
##        .n.fit .n.new.x .n.new.y .n.trn err.abs.OOB.mean err.abs.fit.mean
## .dummy  17841     3772     3772  17841         224248.3         239133.4
##        err.abs.trn.mean err.abs.new.mean.x err.abs.new.mean.y
## .dummy         239133.4           224248.3           224248.3
##        err.abs.OOB.sum err.abs.fit.sum err.abs.trn.sum err.abs.new.sum.x
## .dummy       845864513      4266379442      4266379442         845864513
##        err.abs.new.sum.y
## .dummy         845864513
##     .freqRatio.Fit     .freqRatio.OOB     .freqRatio.Tst 
##                1.0                1.0                1.0 
##             .n.Fit             .n.OOB             .n.Tst 
##            17841.0             3772.0             3772.0 
##             .n.fit           .n.new.x           .n.new.y 
##            17841.0             3772.0             3772.0 
##             .n.trn   err.abs.OOB.mean   err.abs.fit.mean 
##            17841.0           224248.3           239133.4 
##   err.abs.trn.mean err.abs.new.mean.x err.abs.new.mean.y 
##           239133.4           224248.3           224248.3 
##    err.abs.OOB.sum    err.abs.fit.sum    err.abs.trn.sum 
##        845864513.4       4266379442.3       4266379442.3 
##  err.abs.new.sum.x  err.abs.new.sum.y 
##        845864513.4        845864513.4
```

```
## [1] "Final.All.X##rcv#glmnet prediction stats for glbObsNew:"
##                  id max.R.sq.new min.RMSE.new max.Adj.R.sq.new
## 1 All.X##rcv#glmnet  -0.01796882       311717        -0.018509
```

```
##      All.X..rcv.glmnet.imp.x All.X__rcv_glmnet.imp All.X..rcv.glmnet.imp.y
## .pos                     100                   100                     100
##      Final.All.X__rcv_glmnet.imp
## .pos                         100
```

```
## [1] "glbObsNew prediction stats:"
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](Houses_tmplt_files/figure-html/predict.data.new-4.png) 

```
##                   label step_major step_minor label_minor     bgn     end
## 21     predict.data.new         10          0           0  99.325 126.559
## 22 display.session.info         11          0           0 126.559      NA
##    elapsed
## 21  27.234
## 22      NA
```

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                        label step_major step_minor label_minor    bgn
## 21          predict.data.new         10          0           0 99.325
## 20         fit.data.training          9          1           1 79.143
## 15                fit.models          8          0           0 34.901
## 16                fit.models          8          1           1 49.692
## 17                fit.models          8          2           2 61.709
## 1                import.data          1          0           0  9.069
## 2               inspect.data          2          0           0 18.917
## 18                fit.models          8          3           3 72.748
## 14           select.features          7          0           0 31.311
## 3                 scrub.data          2          1           1 26.934
## 10      extract.features.end          3          5           5 29.506
## 13   partition.data.training          6          0           0 30.774
## 19         fit.data.training          9          0           0 78.717
## 11       manage.missing.data          4          0           0 30.391
## 12              cluster.data          5          0           0 30.706
## 9      extract.features.text          3          4           4 29.454
## 6    extract.features.string          3          1           1 29.333
## 4             transform.data          2          2           2 29.276
## 7  extract.features.datetime          3          2           2 29.384
## 8     extract.features.price          3          3           3 29.420
## 5           extract.features          3          0           0 29.314
##        end elapsed duration
## 21 126.559  27.234   27.234
## 20  99.325  20.182   20.182
## 15  49.691  14.790   14.790
## 16  61.708  12.017   12.016
## 17  72.748  11.039   11.039
## 1   18.917   9.848    9.848
## 2   26.933   8.017    8.016
## 18  78.717   5.969    5.969
## 14  34.900   3.590    3.589
## 3   29.275   2.341    2.341
## 10  30.390   0.884    0.884
## 13  31.310   0.537    0.536
## 19  79.142   0.426    0.425
## 11  30.705   0.314    0.314
## 12  30.774   0.068    0.068
## 9   29.506   0.052    0.052
## 6   29.383   0.051    0.050
## 4   29.313   0.038    0.037
## 7   29.420   0.036    0.036
## 8   29.454   0.034    0.034
## 5   29.333   0.019    0.019
## [1] "Total Elapsed Time: 126.559 secs"
```

![](Houses_tmplt_files/figure-html/display.session.info-1.png) 
