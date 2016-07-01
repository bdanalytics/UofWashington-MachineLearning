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

mypltModelStats <- function(df, highLightIx = NULL) {
    bstDf <- tidyr::gather(df[highLightIx, ], 'key', 'value'
                           , -l2Penalty # Remove dims
    )

    pltLst <- list()

    # Print non-facet dims for debugging only ?
    # print("nSteps: ")
    # print(sort(unique(df$nSteps)))

    #         for (num_steps in sort(unique(df$num_steps)))
    #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
    #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
    pltDf <- tidyr::gather(df[
        #                                    (df$num_steps      == num_steps     ) &
        #                                    (df$learnDecayRate == learnDecayRate) &
        #                                    (df$kpRELUs        == kpRELUs       )
        , ], 'key', 'value'
        , -l2Penalty # Remove dims
    )

    #print(nrow(pltDf))
    if (nrow(pltDf) > 0) {
        #print(pltDf)
        #tfSsnLoss OOBAccuracy
        gp <- ggplot(pltDf,
                     aes_string(x = 'l2Penalty', y = 'value'
                                # group for lines
                                #, group = 'nRELUs'
                     )) +
            #geom_line(aes(color = as.factor(nRELUs))) +
            geom_line() +
            # hline if x-axis has log scale & x = 0 value needs to be highlighted
            #                             geom_hline(data = subset(pltDf, (l2Penalty1 == 0)),
            #                                         aes(yintercept = value, color = as.factor(nRELUs)),
            #                                         linetype = 'dashed') +
            geom_point(data = bstDf[(bstDf$key == 'OOBRSS'), ],
                       shape = 5, color = 'black', size = 3) +
            #scale_x_log10() +
            ylab('') +
            scale_linetype_identity(guide = "legend") +
            #guides(linetype = "legend") +
            facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
            #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
            theme(legend.position = "bottom")
        # ggtitle(sprintf('Convolution Neural Net: '))
        pltLst[[1]] <- gp
        #                         pltLst[[paste(as.character(num_steps),
        #                                       as.character(learnDecayRate),
        #                                       as.character(kpRELUs),
        #                                       sep = '#')]] <- gp
    }
    #         }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = 1))
    # dev.off()
}

print(mypltModelStats(df = mdlDf[, setdiff(names(mdlDf), c('.intercept', glbFeats))],
                      highLightIx = which.min(mdlDf$OOBRSS)))

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1)
                #geom_line(aes(color = as.factor(nRELUs))) +
                gp <- gp + geom_line(aes(color = key)) else
                    gp <- gp + geom_line(color = 'blue')

                if (!is.null(scaleXFn) &&
                    !is.null(scaleXFn[dim[1]]))
                    gp <- gp + scaleXFn[dim[1]]

                # hline if x-axis has log scale & x = 0 value needs to be highlighted
                #                             geom_hline(data = subset(pltDf, (l2Penalty1 == 0)),
                #                                         aes(yintercept = value, color = as.factor(nRELUs)),
                #                                         linetype = 'dashed') +
                #scale_x_log10() +

                gp <- gp +
                    ylab('') +
                    scale_linetype_identity(guide = "legend") +
                    #guides(linetype = "legend") +
                    #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                    #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                    theme(legend.position = "bottom")

                if (!is.null(title))
                    gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                        gp <- gp + ggtitle(sprintf(': %s',          key))

                if (!is.null(highLightIx)) {
                    gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                          shape = 5, color = 'black', size = 3)
                }

                pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
                #                         pltLst[[paste(as.character(num_steps),
                #                                       as.character(learnDecayRate),
                #                                       as.character(kpRELUs),
                #                                       sep = '#')]] <- gp
        }
        #         }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

# print(mypltModelStats(df = mdlDf,
#                       measure = c("OOBRSS"),
#                       dim = c("l2Penalty"),
#                       highLightIx = which.min(mdlDf$OOBRSS),
#                       title = NULL))
print(mypltModelStats(df = mdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = scale_x_log10()),
                      highLightIx = which.min(mdlDf$OOBRSS)))

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1)
                #geom_line(aes(color = as.factor(nRELUs))) +
                gp <- gp + geom_line(aes_string(color = dim[2])) else
                    gp <- gp + geom_line(color = 'blue')

                if (!is.null(scaleXFn) &&
                    !is.null(scaleXFn[dim[1]])) {
                    gp <- gp + switch(scaleXFn[dim[1]],
                                      log10 = scale_x_log10(),
                                      stop("switch error in mypltModelStats"))

                    if (scaleXFn[dim[1]] == "log10") {
                        #print("scaleXFn is log10")
                        if (0 %in% unique(df[, dim[1]]))
                            # hline if x-axis has log scale & x = 0 value needs to be highlighted
                            if (length(dim) > 1)
                                gp <- gp +
                                    geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                                (pltDf[, 'key' ] == key) , ],
                                               aes_string(yintercept = "value",  color = dim[2]),
                                               linetype = 'dashed') else
                                                   gp <- gp +
                                                       geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                                                   (pltDf[, 'key' ] == key) , ],
                                                                  aes(yintercept = value), color = 'blue',
                                                                  linetype = 'dashed')
                    }
                }

                gp <- gp +
                    ylab('') +
                    scale_linetype_identity(guide = "legend") +
                    #guides(linetype = "legend") +
                    #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                    #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                    theme(legend.position = "bottom")

                if (!is.null(title))
                    gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                        gp <- gp + ggtitle(sprintf(': %s',          key))
                #                         pltLst[[paste(as.character(num_steps),
                #                                       as.character(learnDecayRate),
                #                                       as.character(kpRELUs),
                #                                       sep = '#')]] <- gp

                if (!is.null(highLightIx)) {
                    gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                          shape = 5, color = 'black', size = 3)
                }

                pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

# print(mypltModelStats(df = mdlDf,
#                       measure = c("OOBRSS"),
#                       dim = c("l2Penalty"),
#                       highLightIx = which.min(mdlDf$OOBRSS),
#                       title = NULL))
print(mypltModelStats(df = mdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(mdlDf$OOBRSS)))

resDf <- foreach(l2PenaltyThs = l2PenaltySearch, .combine = rbind) %do% {
    isPresent <- FALSE
    if ((nrow(mdlDf) > 0) &&
        (nrow(thsDf <- subset(mdlDf, l2Penalty == l2PenaltyThs)) > 0)) {
        thsRes <- NULL
        isPresent <- TRUE
    }

    if (!isPresent) {
        print("")
        print(sprintf(
            "Running optimizeGradientDescent for l2Penalty:%0.4e",
            l2PenaltyThs))
        startTm <- proc.time()["elapsed"]
        mdlWeights <-
            optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                    stepSizeSearch,
                                    l2PenaltyThs,
                                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                                    maxIterationsSearch, verbose = FALSE)
        print('  mdlWeights:')
        print(mdlWeights)
        thsRes <- data.frame(l2Penalty = l2PenaltyThs,
                             elapsedSecs = proc.time()["elapsed"] - startTm)
        thsWeightsDf <- data.frame(matrix(c(mdlWeights,
                                            getObsRSS(glbObsOOB, glbFeats, mdlWeights)),
                                          nrow = 1))
        names(thsWeightsDf) <- c('.intercept', glbFeats, 'OOBRSS')
        thsRes <- cbind(thsRes, thsWeightsDf)
        row.names(thsRes) <- paste(l2PenaltyThs)
    }

    thsRes
}

dimDf <- expand.grid(l2Penalty = l2PenaltySrch, stepSize = stepSizeSrch)
savdimDf <- dimDf

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    if (length(dim) > 2)
        stop(sprintf("mypltModelStats: length(dim): %d exceeds max:%d", length(dim), 2))

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1)
                #geom_line(aes(color = as.factor(nRELUs))) +
                gp <- gp + geom_line(aes_string(color = as.factor(dim[2]))) else
                    gp <- gp + geom_line(color = 'blue')

                if (!is.null(scaleXFn) &&
                    !is.null(scaleXFn[dim[1]])) {
                    gp <- gp + switch(scaleXFn[dim[1]],
                                      log10 = scale_x_log10(),
                                      stop("switch error in mypltModelStats"))

                    if (scaleXFn[dim[1]] == "log10") {
                        #print("scaleXFn is log10")
                        if (0 %in% unique(df[, dim[1]]))
                            # hline if x-axis has log scale & x = 0 value needs to be highlighted
                            if (length(dim) > 1)
                                gp <- gp +
                                    geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                                (pltDf[, 'key' ] == key) , ],
                                               aes_string(yintercept = "value",
                                                          color = as.factor(dim[2])),
                                               linetype = 'dashed') else
                                                   gp <- gp +
                                                       geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                                                   (pltDf[, 'key' ] == key) , ],
                                                                  aes(yintercept = value), color = 'blue',
                                                                  linetype = 'dashed')
                    }
                }

                gp <- gp +
                    ylab('') +
                    scale_linetype_identity(guide = "legend") +
                    #guides(linetype = "legend") +
                    #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                    #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                    theme(legend.position = "bottom")

                if (!is.null(title))
                    gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                        gp <- gp + ggtitle(sprintf(': %s',          key))
                #                         pltLst[[paste(as.character(num_steps),
                #                                       as.character(learnDecayRate),
                #                                       as.character(kpRELUs),
                #                                       sep = '#')]] <- gp

                if (!is.null(highLightIx)) {
                    gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                          shape = 5, color = 'black', size = 3)
                }

                pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

print(mypltModelStats(df = mdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(mdlDf$OOBRSS)))
print(mypltModelStats(df = mdlDf[mdlDf$stepSize == 1e-12, ],
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(mdlDf$OOBRSS)))

dimSrch <- list(l2Penalty = l2PenaltySrch, stepSize = stepSizeSrch)
dimDf <- do.call(expand.grid, dimSrch)
resDf <- foreach(dimSrchIx = 1:nrow(dimDf), .combine = rbind) %do% {
    isPresent <- FALSE
    if ((nrow(mdlDf) > 0) &&
        (nrow(thsDf <- merge(dimDf[dimSrchIx, ], mdlDf, by = names(dimSrch))) > 0)) {
        thsRes <- NULL
        isPresent <- TRUE
    }

    if (!isPresent) {
        print("")
        print(sprintf("Running optimizeGradientDescent for:"))
        for (dim in names(dimSrch))
            print(sprintf("  %*s:%0.4e",
                          max(sapply(names(dimSrch), function(dimNm) nchar(dimNm))),
                          dim, dimDf[dimSrchIx, dim]))
        startTm <- proc.time()["elapsed"]
        mdlWeights <-
            optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                    stepSize  = dimDf[dimSrchIx, 'stepSize' ],
                                    l2Penalty = dimDf[dimSrchIx, 'l2Penalty'],
                                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                                    maxIterationsSrch, verbose = FALSE)
        print('  mdlWeights:')
        print(mdlWeights)
        thsRes <- dimDf[dimSrchIx, ]
        thsRes['elapsedSecs'] <- proc.time()["elapsed"] - startTm
        thsWeightsDf <- data.frame(matrix(c(mdlWeights,
                                            getObsRSS(glbObsOOB, glbFeats, mdlWeights)),
                                          nrow = 1))
        names(thsWeightsDf) <- c('.intercept', glbFeats, 'OOBRSS')
        thsRes <- cbind(thsRes, thsWeightsDf)
        row.names(thsRes) <- do.call(paste, list(dimDf[dimSrchIx, ], collapse = "#"))
    }

    thsRes
}
mdlDf <- rbind(mdlDf, resDf)
print(dplyr::arrange(mdlDf, desc(OOBRSS)))

save(mdlDf, file = mdlDfFlnm)

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    if (length(dim) > 2)
        stop(sprintf("mypltModelStats: length(dim): %d exceeds max:%d", length(dim), 2))

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1) {
                #gp <- gp + geom_line(aes_string(color = as.factor(dim[2])))
                aesStr <- sprintf("color = as.factor(%s)", dim[2])
                aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                gp <- gp + geom_line(mapping = aesMap)
            } else
                gp <- gp + geom_line(color = 'blue')

            if (!is.null(scaleXFn) &&
                !is.null(scaleXFn[dim[1]])) {
                gp <- gp + switch(scaleXFn[dim[1]],
                                  log10 = scale_x_log10(),
                                  stop("switch error in mypltModelStats"))

                if (scaleXFn[dim[1]] == "log10") {
                    #print("scaleXFn is log10")
                    if (0 %in% unique(df[, dim[1]]))
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                }
            }

            gp <- gp +
                ylab('') +
                scale_linetype_identity(guide = "legend") +
                #guides(linetype = "legend") +
                #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                theme(legend.position = "bottom")

            if (!is.null(title))
                gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                    gp <- gp + ggtitle(sprintf(': %s',          key))
            #                         pltLst[[paste(as.character(num_steps),
            #                                       as.character(learnDecayRate),
            #                                       as.character(kpRELUs),
            #                                       sep = '#')]] <- gp

            if (!is.null(highLightIx)) {
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)
            }

            pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

pltMdlDf <- mdlDf[(mdlDf$OOBRSS <= 1e+273), ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[  (mdlDf$OOBRSS   <= 1e+273) &
                        (mdlDf$stepSize == 1e-11 )
                    , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

gradientFnSrch <- c(ridgeRegressionLossGradientFn, ridgeRegressionLossGradientFn)
#maxIterationsSrch <- c(100, 200)
maxIterationsSrch <- c(100)
#stepSizeSrch <- c(1e-13, 1e-12, 1e-11, 2e-11)
stepSizeSrch <- c(1e-11)
#l2PenaltySrch <- c(0e+0, 1e+6, 1e+8, 1e+10, 1e+12)
l2PenaltySrch <- c(0e+0)

dimSrch <- list(l2Penalty = l2PenaltySrch        ,
                stepSize  = stepSizeSrch         ,
                maxIterations = maxIterationsSrch)
dimDf <- do.call(expand.grid, dimSrch)
resDf <- foreach(dimSrchIx = 1:nrow(dimDf), .combine = rbind) %do% {
    isPresent <- FALSE
    if ((nrow(mdlDf) > 0) &&
        (nrow(thsDf <- merge(dimDf[dimSrchIx, ], mdlDf, by = names(dimSrch))) > 0)) {
        thsRes <- NULL
        isPresent <- TRUE
    }

    if (!isPresent) {
        print("")
        print(sprintf("Running optimizeGradientDescent for:"))
        for (dim in names(dimSrch))
            print(sprintf("  %*s:%0.4e",
                          max(sapply(names(dimSrch), function(dimNm) nchar(dimNm))),
                          dim, dimDf[dimSrchIx, dim]))
        startTm <- proc.time()["elapsed"]
        mdlWeights <-
            optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                    stepSize  = dimDf[dimSrchIx, 'stepSize' ],
                                    l2Penalty = dimDf[dimSrchIx, 'l2Penalty'],
                                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                                    maxIterations = dimDf[dimSrchIx, 'maxIterations'],
                                    verbose = FALSE)
        print('  mdlWeights:')
        print(mdlWeights)
        thsRes <- dimDf[dimSrchIx, ]
        thsRes['elapsedSecs'] <- proc.time()["elapsed"] - startTm
        thsWeightsDf <- data.frame(matrix(c(mdlWeights,
                                            getObsRSS(glbObsOOB, glbFeats, mdlWeights)),
                                          nrow = 1))
        names(thsWeightsDf) <- c('.intercept', glbFeats, 'OOBRSS')
        thsRes <- cbind(thsRes, thsWeightsDf)
        row.names(thsRes) <- do.call(paste, list(dimDf[dimSrchIx, ], collapse = "#"))
    }

    thsRes
}
mdlDf <- rbind(mdlDf, resDf)
print(dplyr::arrange(mdlDf, desc(OOBRSS)))

save(mdlDf, file = mdlDfFlnm)

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    if (length(dim) > 3)
        stop(sprintf("mypltModelStats: length(dim): %d exceeds max:%d", length(dim), 3))

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1) {
                #gp <- gp + geom_line(aes_string(color = as.factor(dim[2])))
                aesStr <- sprintf("color = as.factor(%s)", dim[2])
                aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                gp <- gp + geom_line(mapping = aesMap)
            } else
                gp <- gp + geom_line(color = 'blue')

            if (!is.null(scaleXFn) &&
                !is.null(scaleXFn[dim[1]])) {
                gp <- gp + switch(scaleXFn[dim[1]],
                                  log10 = scale_x_log10(),
                                  stop("switch error in mypltModelStats"))

                if (scaleXFn[dim[1]] == "log10") {
                    #print("scaleXFn is log10")
                    if (0 %in% unique(df[, dim[1]]))
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                }
            }

            gp <- gp +
                ylab('') +
                scale_linetype_identity(guide = "legend") +
                #guides(linetype = "legend") +
                #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                theme(legend.position = "bottom")

            if (length(dim) == 3)
                gp <- gp + facet_grid(as.formula(paste(dim[3], "~ .")),
                                      scales = "free", labeller = "label_both")

            if (!is.null(title))
                gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                    gp <- gp + ggtitle(sprintf(': %s',          key))
            #                         pltLst[[paste(as.character(num_steps),
            #                                       as.character(learnDecayRate),
            #                                       as.character(kpRELUs),
            #                                       sep = '#')]] <- gp

            if (!is.null(highLightIx)) {
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)
            }

            pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS <= 1e+273), ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS        <= 1e+273) &
                          (mdlDf$maxIterations == 100   )
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[  (mdlDf$OOBRSS        <= 1e+273) &
                        (mdlDf$maxIterations == 100   ) &
                        (mdlDf$stepSize      == 1e-11 )
                    , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    # if (length(dim) > 4)
    #     stop(sprintf("mypltModelStats: length(dim): %d exceeds max:%d", length(dim), 3))

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        # Print non-facet dims for debugging only ?
        # print("nSteps: ")
        # print(sort(unique(df$nSteps)))

        #         for (num_steps in sort(unique(df$num_steps)))
        #             for (learnDecayRate in sort(unique(df$learnDecayRate)))
        #                 for (kpRELUs in sort(unique(df$kpRELUs))) {
        pltDf <- tidyr::gather_(df[
            #                                    (df$num_steps      == num_steps     ) &
            #                                    (df$learnDecayRate == learnDecayRate) &
            #                                    (df$kpRELUs        == kpRELUs       )
            , c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        #print(nrow(pltDf))
        if (nrow(pltDf) > 0) {
            #print(pltDf)
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1) {
                #gp <- gp + geom_line(aes_string(color = as.factor(dim[2])))
                aesStr <- sprintf("color = as.factor(%s)", dim[2])
                aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                gp <- gp + geom_line(mapping = aesMap)
            } else
                gp <- gp + geom_line(color = 'blue')

            if (!is.null(scaleXFn) &&
                !is.null(scaleXFn[dim[1]])) {
                gp <- gp + switch(scaleXFn[dim[1]],
                                  log10 = scale_x_log10(),
                                  stop("switch error in mypltModelStats"))

                if (scaleXFn[dim[1]] == "log10") {
                    #print("scaleXFn is log10")
                    if (0 %in% unique(df[, dim[1]]))
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                }
            }

            gp <- gp +
                ylab('') +
                scale_linetype_identity(guide = "legend") +
                #guides(linetype = "legend") +
                #facet_grid('key ~ .', labeller = label_both, scales = 'free_y') +
                #facet_grid('key ~ l2Penalty3', labeller = label_both, scales = 'free_y') +
                theme(legend.position = "bottom")

            if (length(dim) >= 3)
                gp <- gp + facet_grid(as.formula(paste(paste0(tail(dim, -2), collapse = "+"),
                                                       "~ .")),
                                      scales = "free", labeller = "label_both")

            if (!is.null(title))
                gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                    gp <- gp + ggtitle(sprintf(': %s',          key))
            #                         pltLst[[paste(as.character(num_steps),
            #                                       as.character(learnDecayRate),
            #                                       as.character(kpRELUs),
            #                                       sep = '#')]] <- gp

            if (!is.null(highLightIx)) {
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)
            }

            pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    # png(filename = '4_convolutions_mdlNSteps.png',
    #     width = 480 * 1, height = 480 * 1)
    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
    # dev.off()
}

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS <= 1e+273), ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations", "gradientFnNm"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS       <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn")
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS        <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                          (mdlDf$maxIterations == 100   )
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[  (mdlDf$OOBRSS        <= 1e+273) &
                        (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                        (mdlDf$maxIterations == 100   ) &
                        (mdlDf$stepSize      == 1e-11 )
                    , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

gradientFnNmSrch <- c("ridgeRegressionLossGradientFn", "autoLossGradientFn")
#maxIterationsSrch <- c(100, 200)
maxIterationsSrch <- c(100)
#stepSizeSrch <- c(1e-13, 1e-12, 1e-11, 2e-11)
stepSizeSrch <- c(1e-11)
#l2PenaltySrch <- c(0e+0, 1e+6, 1e+8, 1e+10, 1e+12)
l2PenaltySrch <- c(0e+0)

dimSrch <- list(l2Penalty     = l2PenaltySrch    ,
                stepSize      = stepSizeSrch     ,
                maxIterations = maxIterationsSrch,
                gradientFnNm  = gradientFnNmSrch )
dimDf <- do.call(expand.grid, dimSrch)
resDf <- foreach(dimSrchIx = 1:nrow(dimDf), .combine = rbind) %do% {
    isPresent <- FALSE
    if ((nrow(mdlDf) > 0) &&
        (nrow(thsDf <- merge(dimDf[dimSrchIx, ], mdlDf, by = names(dimSrch))) > 0)) {
        thsRes <- NULL
        isPresent <- TRUE
    }

    if (!isPresent) {
        print("")
        print(sprintf("Running optimizeGradientDescent for:"))
        for (dim in names(dimSrch))
            print(sprintf(ifelse(is.numeric(dimDf[, dim]), "  %*s:%0.4e", "  %*s:%s"),
                          max(sapply(names(dimSrch), function(dimNm) nchar(dimNm))),
                          dim, dimDf[dimSrchIx, dim]))
        startTm <- proc.time()["elapsed"]
        mdlWeights <-
            optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                    stepSize  = dimDf[dimSrchIx, 'stepSize' ],
                                    l2Penalty = dimDf[dimSrchIx, 'l2Penalty'],
                                    ridgeRegressionLossFn,
                                    switch(as.character(dimDf[dimSrchIx, dim]),
                                           "ridgeRegressionLossGradientFn" = ridgeRegressionLossGradientFn,
                                           "autoLossGradientFn" =            autoLossGradientFn,
                                           "default" = stop("unknown gradientFnNm")),
                                    maxIterations = dimDf[dimSrchIx, 'maxIterations'],
                                    verbose = FALSE)
        print('  mdlWeights:')
        print(mdlWeights)
        thsRes <- dimDf[dimSrchIx, ]
        thsRes['elapsedSecs'] <- proc.time()["elapsed"] - startTm
        thsWeightsDf <- data.frame(matrix(c(mdlWeights,
                                            getObsRSS(glbObsOOB, glbFeats, mdlWeights)),
                                          nrow = 1))
        names(thsWeightsDf) <- c('.intercept', glbFeats, 'OOBRSS')
        thsRes <- cbind(thsRes, thsWeightsDf)
        row.names(thsRes) <- do.call(paste, list(dimDf[dimSrchIx, ], collapse = "#"))
    }

    thsRes
}
mdlDf <- rbind(mdlDf, resDf)
print(dplyr::arrange(mdlDf, desc(OOBRSS)))

save(mdlDf, file = mdlDfFlnm)

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL, fileName = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    df <- df[, c(measure, dim)]

    pltLst <- list(); pltIx <- 1

    for (key in measure) {
        pltDf <- tidyr::gather_(df[, c(key, dim)], 'key', 'value', gather_cols = key)
        if (!is.null(highLightIx))
            bstDf <- tidyr::gather_(df[highLightIx, c(key, dim)], 'key', 'value',
                                    gather_cols = key)

        if (nrow(pltDf) > 0) {
            gp <- ggplot(pltDf,
                         aes_string(x = dim[1], y = 'value'
                                    # group for lines
                                    #, group = 'nRELUs'
                         ))

            if (length(dim) > 1) {
                aesStr <- sprintf("color = as.factor(%s)", dim[2])
                aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                gp <- gp + geom_line(mapping = aesMap)
            } else
                gp <- gp + geom_line(color = 'blue')

            if (!is.null(scaleXFn) &&
                !is.null(scaleXFn[dim[1]])) {
                gp <- gp + switch(scaleXFn[dim[1]],
                                  log10 = scale_x_log10(),
                                  stop("switch error in mypltModelStats"))

                if (scaleXFn[dim[1]] == "log10") {
                    #print("scaleXFn is log10")
                    if (0 %in% unique(df[, dim[1]]))
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                }
            }

            gp <- gp +
                ylab('') +
                scale_linetype_identity(guide = "legend") +
                theme(legend.position = "bottom")

            if (length(dim) >= 3)
                gp <- gp + facet_grid(as.formula(paste(paste0(tail(dim, -2), collapse = "+"),
                                                       "~ .")),
                                      scales = "free", labeller = "label_both")

            if (!is.null(title))
                gp <- gp + ggtitle(sprintf('%s: %s', title, key)) else
                    gp <- gp + ggtitle(sprintf(': %s',          key))

            if (!is.null(highLightIx)) {
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)
            }

            pltLst[[pltIx]] <- gp; pltIx <- pltIx + 1
        }
    }

    if (!is.null(fileName)) {
        savPltLst <- pltLst
        png(filename = fileName, width = 480 * 1, height = 480 * 1)
        mypltMultiple(plotlist = pltLst, cols = length(measure))
        dev.off()

        pltLst <- savPltLst
    }

    return(mypltMultiple(plotlist = pltLst, cols = length(measure)))
}

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS <= 1e+273), ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations", "gradientFnNm"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS']),
                      fileName = 'WAKCHouses_SGD_Test_r.png'))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS       <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn")
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS        <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                          (mdlDf$maxIterations == 100   )
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[  (mdlDf$OOBRSS        <= 1e+273) &
                        (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                        (mdlDf$maxIterations == 100   ) &
                        (mdlDf$stepSize      == 1e-11 )
                    , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

mypltModelStats <- function(df, measure, dim = NULL, scaleXFn = NULL, highLightIx = NULL,
                            title = NULL, fileName = NULL) {
    if (is.null(dim))
        dim <- setdiff(names(df), measure)

    df <- df[, c(measure, dim)]

    pltDf <- tidyr::gather_(df, 'key', 'value', gather_cols = measure)
    if (!is.null(highLightIx))
        bstDf <- tidyr::gather_(df[highLightIx, ], 'key', 'value',
                                gather_cols = measure)

    if (nrow(pltDf) > 0) {
        gp <- ggplot(pltDf, aes_string(x = dim[1], y = 'value'))

        if (length(dim) > 1) {
            aesStr <- sprintf("color = as.factor(%s)", dim[2])
            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
            gp <- gp + geom_line(mapping = aesMap)
        } else
            gp <- gp + geom_line(color = 'blue')

        if (!is.null(scaleXFn) &&
            !is.null(scaleXFn[dim[1]])) {
            gp <- gp + switch(scaleXFn[dim[1]],
                              log10 = scale_x_log10(),
                              stop("switch error in mypltModelStats"))

            if (scaleXFn[dim[1]] == "log10") {
                #print("scaleXFn is log10")
                if (0 %in% unique(df[, dim[1]]))
                    for (key in measure) {
                        # hline if x-axis has log scale & x = 0 value needs to be highlighted
                        if (length(dim) > 1) {
                            aesStr <-
                                sprintf("yintercept = value, color = as.factor(%s)", dim[2])
                            aesMap <- eval(parse(text = paste("aes(", aesStr, ")")))
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           mapping = aesMap,
                                           linetype = 'dashed')
                        } else
                            gp <- gp +
                                geom_hline(data = pltDf[(pltDf[, dim[1]] == 0  ) &
                                                            (pltDf[, 'key' ] == key) , ],
                                           aes(yintercept = value), color = 'blue',
                                           linetype = 'dashed')
                    }
            }
        }

        gp <- gp +
            ylab('') +
            scale_linetype_identity(guide = "legend") +
            theme(legend.position = "bottom")

        if (length(dim) < 3)
            gp <- gp + facet_grid(. ~ key,
                                  scales = "free", labeller = "label_both")
        gp <- gp + facet_grid(as.formula(paste(paste0(tail(dim, -2), collapse = "+"),
                                               "~ key")),
                              scales = "free", labeller = "label_both")

        if (!is.null(title))
            gp <- gp + ggtitle(title)

        if (!is.null(highLightIx))
            for (key in measure)
                gp <- gp + geom_point(data = bstDf[(bstDf$key == key), ],
                                      shape = 5, color = 'black', size = 3)

    }

    if (!is.null(fileName)) {
        savGP <- gp
        png(filename = fileName, width = 480 * 1, height = 480 * 1)
        print(gp)
        dev.off()

        gp <- savGP
    }

    return(gp)
}

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS <= 1e+273), ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations", "gradientFnNm"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS']),
                      fileName = 'WAKCHouses_SGD_Test_r.png'))
print(mypltModelStats(df = pltMdlDf,
                      measure = c("elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations", "gradientFnNm"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS']),
                      fileName = NULL))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS       <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn")
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize", "maxIterations"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[    (mdlDf$OOBRSS        <= 1e+273) &
                          (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                          (mdlDf$maxIterations == 100   )
                      , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty", "stepSize"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

pltMdlDf <- mdlDf[  (mdlDf$OOBRSS        <= 1e+273) &
                        (mdlDf$gradientFnNm == "ridgeRegressionLossGradientFn") &
                        (mdlDf$maxIterations == 100   ) &
                        (mdlDf$stepSize      == 1e-11 )
                    , ]
print(mypltModelStats(df = pltMdlDf,
                      measure = c("OOBRSS", "elapsedSecs"),
                      dim = c("l2Penalty"),
                      scaleXFn = c(l2Penalty = "log10"),
                      highLightIx = which.min(pltMdlDf[, 'OOBRSS'])))

