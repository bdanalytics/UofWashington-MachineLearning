
print(format(Sys.time(), "(%a) %b %d, %Y"))

require(doMC)
glbCores <- 6 # of cores on machine - 2
registerDoMC(glbCores)

require(Cairo)
#require(dplyr)
#require(ggplot2)
#require(jpeg)
#require(reshape2)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")

glbObsTrnFile <- list(name = "kc_house_train_data.csv")
glbObsNewFile <- list(name = "kc_house_test_data.csv")
glb_rsp_var_raw <- "price"
glb_rsp_var <- glb_rsp_var_raw
glbObsTrnPartitionSeed <- 123

glbFeats <- c('sqft_living', 'sqft_living15')

glbObsTrn <- myimport_data(specs = glbObsTrnFile, comment = "glbObsTrn",
                           force_header = TRUE)

glbObsNew <- myimport_data(specs = glbObsNewFile, comment = "glbObsNew",
                           force_header = TRUE)

set.seed(glbObsTrnPartitionSeed)
OOB_size <- nrow(glbObsNew) * 1.1
print(sprintf("Fit vs. OOB split ratio:%0.4f",
              1 - (OOB_size * 1.0 / nrow(glbObsTrn))))

require(caTools)
split <- sample.split(glbObsTrn[, glb_rsp_var_raw], SplitRatio = OOB_size)
print(sum(split))
glbObsOOB <- glbObsTrn[ split, ]
glbObsFit <- glbObsTrn[!split, ]
print(sprintf("glbObsFit:")); print(dim(glbObsFit))
print(sprintf("glbObsOOB:")); print(dim(glbObsOOB))

print(OOB_size)
print(OOB_size * 1.0 / nrow(glbObsTrn))

weightsZero <- rep(0, 1 + length(glbFeats))
print(sprintf("weightsZero:"))
print(weightsZero)

predictOutput <- function(obsDf, feats, weights) {
    featMtrx <- cbind(matrix(rep(1.0, nrow(obsDf)), nrow = nrow(obsDf)),
                      as.matrix(obsDf[, feats]))
    #print(class(featMtrx))
    return(featMtrx %*% weights)
}

print(predictOutput(glbObsTrn, 'sqft_living', c(1.0, 1.0))[1]) # should be 1181.0
print(predictOutput(glbObsTrn, 'sqft_living', c(1.0, 1.0))[2]) # should be 2571.0

print(predictOutput(glbObsTrn, 'sqft_living', c(10.0, 2.0))[1]) # should be 2370.0
print(predictOutput(glbObsTrn, 'sqft_living', c(10.0, 2.0))[2]) # should be 5150.0

ridgeRegressionLossFn <- function(obsDf, feats, weights, l2Penalty) {
    loss <- sum((predictOutput(obsDf, feats, weights) - obsDf[, glb_rsp_var])
                ^ 2) +
                l2Penalty * sum(weights ^ 2)
    if (is.infinite(loss)) {
        print(sprintf("ridgeRegressionLossFn: loss == Inf; l2Penalty:%0.4e; weights:", l2Penalty))
        print(weights)
    }
    return(loss)
}

ridgeRegressionLossGradientFn <- function(obsDf, feats, weights, l2Penalty,
                                          featIx, isIntercept) {
#     print(sprintf(
#         "ridgeRegressionLossGradientFn: nrow(obsDf):%d; l2Penalty:%0.4f; featIx:%d; isIntercept:%s",
#                       nrow(obsDf), l2Penalty, featIx, isIntercept))
#     print(sprintf(
#         "ridgeRegressionLossGradientFn: weights:"))
#     print(weights)

    if (!isIntercept) {
        featX <- as.matrix(obsDf[, feats[featIx-1]])
    } else featX <- matrix(rep(1, nrow(obsDf)), nrow = nrow(obsDf))

    gradient <- 2 *
                sum((predictOutput(obsDf, feats, weights) -
                     obsDf[, glb_rsp_var]) *
                    featX)
    if (!isIntercept)
        gradient <- gradient + 2 * l2Penalty * weights[featIx]

#     print(sprintf(
#         "ridgeRegressionLossGradientFn: featIx:%d; gradient:%0.4e",
#                   featIx, gradient))
    return(gradient)
}

example_weights = c(1.0, 10.0)
example_predictions = predictOutput(glbObsTrn, 'sqft_living', example_weights)
example_errors = example_predictions - glbObsTrn[, glb_rsp_var]

# next two lines should print the same values
print(sum(example_errors * glbObsTrn[, 'sqft_living'])*2+20)
# print(example_errors[1:5])
# print(glbObsTrn[1:5, 'sqft_living'])
# print((example_errors * glbObsTrn[, 'sqft_living'])[1:5])
print(ridgeRegressionLossGradientFn(glbObsTrn, 'sqft_living',
                                    example_weights,
                                    l2Penalty = 1, featIx = 2, FALSE))

# next two lines should print the same values
print('')
print(sum(example_errors)*2)
#print(example_errors[1:5])
print(ridgeRegressionLossGradientFn(glbObsTrn, 'sqft_living',
                                    example_weights,
                                    l2Penalty = 1, featIx = 1, TRUE))

optimizeGradientDescent <- function(obsDf, feats, weightsInitial,
                                    stepSize, l2Penalty,
                                    lossFn, lossGradientFn,
                                    maxIterations = 100, verbose = FALSE,
                                    maxLoss = 1e156) {

    if (verbose) {
        print(" ")
        print(sprintf("optimzeGradientDescent:"))
    }
    weights <- weightsInitial
    loss <- lossFn(obsDf, feats, weights, l2Penalty)

    #while not reached maximum number of iterations:
    iterResults = data.frame(iterNum = 1:maxIterations)
    for (iterNum in 1:maxIterations) {
        if (verbose &&
            ((iterNum %% (maxIterations / 10) == 1) ||
             (iterNum <=10)))
            print(sprintf("  iteration: %d; loss:%0.4e", iterNum, loss))

        # loop over each weight
        for (i in 1:length(weights)) {
            # compute the derivative for weight[i].
            #  when i=1, you are computing the derivative of the constant!
            gradient <-
                lossGradientFn(obsDf, feats, weights, l2Penalty,
                               i, (i == 1))

            # subtract the stepSize times the gradient from the
            #   current weight
            weights[i] = weights[i] - stepSize * gradient
        }

        loss <- lossFn(obsDf, feats, weights, l2Penalty)
        iterResults[iterNum, "loss"] <- loss
        for (featIx in 1:(length(feats) + 1)) {
            if (featIx == 1)
                iterResults[iterNum, '.intercept'] <- weights[featIx] else
                iterResults[iterNum, feats[featIx - 1]] <- weights[featIx]
        }

        if (is.infinite(loss)) {
            warning("optimizeGradientDescent: loss is Inf")
            break
        }
    }

    if ((sum(      is.na(iterResults$loss)) > 0) ||
        (sum(is.infinite(iterResults$loss)) > 0))
        converged <- FALSE else converged <- TRUE

    # Display results at end of iterations
    if (verbose || !converged) {
        myprint_df(iterResults)

        iterResultsIx <- ifelse(converged, nrow(iterResults),
                                            which(      is.na(iterResults$loss) |
                                                  is.infinite(iterResults$loss))[1] - 1)
        if (max(iterResults[1:iterResultsIx, 'loss']) > maxLoss) # geom_contour does not work
            iterResultsIx <- which(iterResults[1:iterResultsIx, 'loss'] > maxLoss)[1] - 1

        print(sprintf('iterResultsIx:%d', iterResultsIx))
        wgt1 <- iterResults[1:iterResultsIx, '.intercept']
        wgt2 <- iterResults[1:iterResultsIx, feats[1]]
        contourDf <- expand.grid(
            wgt1 = union(rnorm(10,
                               mean(wgt1, na.rm = TRUE),
                               sd(wgt1, na.rm = TRUE)),
                         union(+1 * range(wgt1, na.rm = TRUE),
                               +2 * range(wgt1, na.rm = TRUE))),
            wgt2 = union(rnorm(10,
                               mean(wgt2, na.rm = TRUE),
                               sd(wgt2, na.rm = TRUE)),
                         union(+1 * range(wgt2, na.rm = TRUE),
                               +2 * range(wgt2, na.rm = TRUE))))
        # for loss == Infinite
        #contourDf <- expand.grid(wgt1 = wgt1[1:20],wgt2 = wgt2[1:20])
        #contourDf[, 'fitLoss'] <- sapply(1:nrow(contourDf), function(contourIx) lossFn(obsDf, feats, c(contourDf[contourIx, 'wgt1'], contourDf[contourIx, 'wgt2'], iterResults[1, 5:length(names(iterResults))]), l2Penalty))

        contourDf[, 'loss'] <-
            sapply(1:nrow(contourDf), function(contourIx)
                lossFn(obsDf, feats, c(contourDf[contourIx, 'wgt1'],
                                       contourDf[contourIx, 'wgt2'],
                                       iterResults[iterResultsIx, 5:length(names(iterResults))]),
                       l2Penalty))
        #print(str(contourDf))
        print(contourDf)
        print(gp <- ggplot(contourDf, aes(x = wgt1, y = wgt2)) +
                  geom_contour(aes(z = loss, color = ..level..)) +
                  geom_path(data = iterResults,
                            aes_string(x = ".intercept", y = feats[1]),
                            color = 'red', lineend = "square") +
                  geom_point(data = iterResults,
                             aes_string(x = '.intercept', y = feats[1]),
                             color = 'black', shape = 4) +
                  geom_point(data = iterResults[1,],
                             aes_string(x = '.intercept', y = feats[1]),
                             color = 'red', shape = 1, size = 5) +
                  xlab('.intercept') + ylab(feats[1])
        )

        print(myplot_line(iterResults[1:iterResultsIx,], "iterNum", "loss"))
    }
    return(weights)
}

stepSize <- 5e-11; l2Penalty <- 1e+10; maxIterations = 100
weightsTst <-
    optimizeGradientDescent(obsDf = glbObsFit, feats = glbFeats,
                            weightsInitial = weightsZero,
                            stepSize = stepSize,
                            l2Penalty = l2Penalty,
                            lossFn = ridgeRegressionLossFn,
                            lossGradientFn = ridgeRegressionLossGradientFn,
                            maxIterations = maxIterations,
                            verbose = TRUE,
                            maxLoss = 1e155)
print(sprintf('weightsTst:'))
print(weightsTst)

stepSize <- 1e-12; l2Penalty <- 0.0; maxIterations = 100
weightsL2Zero <-
    optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                            stepSize, l2Penalty,
                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                            maxIterations, verbose = TRUE)
print(sprintf('weightsL2Zero:'))
print(weightsL2Zero)

stepSize <- 1e-12; l2Penalty <- 1e10; maxIterations = 100
weightsL2Hgh <-
    optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                            stepSize, l2Penalty,
                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                            maxIterations, verbose = TRUE)
print(sprintf('weightsL2Hgh:'))
print(weightsL2Hgh)

stepSize <- 1e-10; l2Penalty <- 1e+10; maxIterations = 100
weightsTest <-
    optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                            stepSize, l2Penalty,
                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                            maxIterations, verbose = TRUE)
print(sprintf('weightsTest:'))
print(weightsTest)

getObsRSS <- function(obsDf, feats, weights) {
    return(sum((obsDf[, glb_rsp_var] -
                predictOutput(obsDf, feats, weights)) ^ 2))
}

maxIterations <- 100
mdlLst <- list()
# stepSize <- 1e-12;
# l2Penalty <- 1e10;
mdlDf <- expand.grid(stepSize  = c(1e-13, 1e-12, 1e-11, 1e-10),
                     l2Penalty = c(1e+02, 1e+04, 1e+06, 1e+08, 1e10))
#print(class(mdlDf))
for (mdlIx in 1:nrow(mdlDf)) {
    print("")
    print(sprintf(
        "Running optimizeGradientDescent for l2Penalty:%0.4e",
          mdlDf[mdlIx, 'l2Penalty']))
    mdlWeights <-
        optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                mdlDf[mdlIx, 'stepSize'],
                                mdlDf[mdlIx, 'l2Penalty'],
                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                                maxIterations, verbose = FALSE)
    print('  mdlWeights:')
    print(mdlWeights)
    for (featIx in 1:(length(glbFeats) + 1)) {
        if (featIx == 1)
            mdlDf[mdlIx, 'intercept'] <- mdlWeights[featIx] else
            mdlDf[mdlIx, glbFeats[featIx - 1]] <- mdlWeights[featIx]
#     mdlDf[mdlIx, c('intercept', glbFeats)] <- mdlWeights
    }
    mdlDf[mdlIx, 'OOBRSS'] <- getObsRSS(glbObsOOB, glbFeats, mdlWeights)
}
# print(sprintf('weightsL2Hgh:'))
# print(weightsL2Hgh)
print(dplyr::arrange(mdlDf, desc(OOBRSS)))

maxIterations <- 100
mdlLst <- list()
mdlDf <- expand.grid(stepSize  = c(1e-13, 1e-12, 1e-11),
                     l2Penalty = c(1e+02, 1e+04, 1e+06, 1e+08, 1e10))
# mdlDf <- expand.grid(stepSize  = c(1e-13, 1e-12),
#                      l2Penalty = c(1e+02, 1e+04))
resDf <- foreach(mdlIx = 1:nrow(mdlDf), .combine = rbind) %do% {
    print("")
    print(sprintf(
        "Running optimizeGradientDescent for l2Penalty:%0.4e",
          mdlDf[mdlIx, 'l2Penalty']))
    mdlWeights <-
        optimizeGradientDescent(glbObsFit, glbFeats, weightsZero,
                                mdlDf[mdlIx, 'stepSize'],
                                mdlDf[mdlIx, 'l2Penalty'],
                    ridgeRegressionLossFn, ridgeRegressionLossGradientFn,
                                maxIterations, verbose = FALSE)
    print('  mdlWeights:')
    print(mdlWeights)
    thsRes <- data.frame(matrix(c(mdlWeights,
                           getObsRSS(glbObsOOB, glbFeats, mdlWeights)),
                               nrow = 1))
    row.names(thsRes) <- mdlIx
    thsRes
}
names(resDf) <- c('.intercept', glbFeats, 'OOBRSS')
mdlDf <- cbind(mdlDf, resDf)
print(dplyr::arrange(mdlDf, desc(OOBRSS)))

print(ggplot(mdlDf, aes(x = l2Penalty, y = OOBRSS, group = stepSize)) +
      geom_line(aes(color = as.factor(stepSize))) +
      scale_x_log10() +
      geom_point(data = mdlDf[which.min(mdlDf$OOBRSS), ],
                 shape = 5, color = 'black', size = 2))

print(" ")
print(sprintf("weightsZero:"))
print(weightsZero)
print(sprintf("  glbObsNew RSS: %.4e",
              getObsRSS(glbObsNew, glbFeats, weightsZero)))

print(" ")
print(sprintf('weightsL2Zero:'))
print(weightsL2Zero)
print(sprintf('  glbObsNew RSS: %.4e',
              getObsRSS(glbObsNew, glbFeats, weightsL2Zero)))

print(sprintf("glbObsNew Obs 1 %s:%0.4f",
              glb_rsp_var, glbObsNew[1, glb_rsp_var]))

print(" ")
print(sprintf("  weightsZero   %s prediction:%0.4f; error.abs:%0.4f",
              glb_rsp_var,
              prediction <- predictOutput(glbObsNew[1, ], glbFeats, weightsZero  ),
              abs(prediction - glbObsNew[1, glb_rsp_var])))

print(" ")
print(sprintf("  weightsL2Zero %s prediction:%0.4f; error.abs:%0.4f",
              glb_rsp_var,
              prediction <- predictOutput(glbObsNew[1, ], glbFeats, weightsL2Zero),
              abs(prediction - glbObsNew[1, glb_rsp_var])))

print(sessionInfo())




