setwd("~/Documents/R ")
library(gbm)
library(randomForest)
library(neuralnet)
library(caret)
library('e1071')
library(glmulti)
library(MuMIn)

        ######################################
        ## HW 1.1 Using boston housing data ##
        ######################################
##  import data

header<-c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")

housing<-data.matrix(read.table("housing.data",col.names=header))

# separated by training and testing
trnx<-housing[1:253,1:13]
trny<-housing[1:253,14]

tstx<-housing[254:506,1:13]
tsty<-housing[254:506,14]

# compute normalization factor
normfact <- function (x) {
  return(sqrt(mean(x^2)))
}


#with scale
normalize <- function (x, nf) {
  for (i in 1:length(nf)) {
    x[,i]=x[,i]/nf[i]
  }
  return(x)
}

# centering
centering <- function (x, mu) {
  if (is.vector(x)) {
    for (i in 1:length(x)) {
      x[i]=x[i]-mu;
    }
  }
  else {
    for (i in 1:dim(x)[1]) {
      x[i,]=x[i,]-mu;
    }
  }
  return(x)
}

#normalization
nfact=apply(trnx,2,normfact)
ntrnx=normalize(trnx,nfact)
ntstx=normalize(tstx,nfact)

#centering
mux=apply(ntrnx,2,mean)
muy=mean(trny)
ntrnx=centering(ntrnx,mux)
ntrny=centering(trny,muy)
ntstx=centering(ntstx,mux)
ntsty=centering(tsty,muy)

# MSE (mean squared error) between predicted-y py and true-y ty
mse <- function(py,ty) {
  return(mean((py-ty)^2))
}

# compute ridge regression weight
myridge.fit <- function(X,y,lambda) {
  w= solve((t(X) %*% X) +(lambda*diag(dim(X)[2])), (t(X) %*% y))
  return(w)
}


# compute predicted Y with coefficients w and data matrix X
predict.linear <- function (w,X) {
  n<-dim(X)[1]
  y<-c(array(0,dim=c(n,1)))
  for (i in 1:n) {
    y[i]=sum(w*X[i,])
  }
  return(y)
}

# compute mean squared error for linear weight w on data (X,y)
mse.linear <- function(w,X,y) {
  py=predict.linear(w,X)
  return (mse(py,y))
}

################
#without scale
trn<-list(x=trnx,y=trny)
#with scale
ntrn<-list(x=ntrnx,y=ntrny)


### Part 1:gbm 
## Method 1: using 70/30 split cv to tune the parameter
# random split according to training ratio
partition.ind <- function(n, ratio)
{
  train.indices= which(runif(n) < ratio)
  return (train.indices)
}

# partition data into training/validation sets
partition.cv <- function(dat, ratio)
{
  n=length(dat$y)
  trn.ind= partition.ind(n,ratio)
  trn=list(x=dat$x[trn.ind,],y=dat$y[trn.ind])
  val=list(x=dat$x[-trn.ind,],y=dat$y[-trn.ind])
  cv=list(train=trn,validation=val)
  return(cv)
}
# Tuning Parameter using cv

tuning.gbm<- function(trn,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      model=gbm.fit(train$x,train$y,distribution="gaussian",n.trees = 300,shrinkage=0.01,interaction.depth=cvec[j],n.minobsinnode = 10,verbose = FALSE)
      pred=predict.gbm(model,data.frame(validation$x),n.trees=300)
      err[i,j]= mean((pred-validation$y)^2)
    }
  }
  
  for (j in 1:nc) {
    print(paste("depth=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}
depth=c(1,2,3,4,5)
tuning.gbm(trn,depth,20,0.7)
#normalized data
tuning.gbm(ntrn,depth,20,0.7)

#
## Method 2: using caret package
library(caret)
fitControl <- trainControl(
  method = "LGOCV",
  p = 0.7)
# Create the tuning parameter intervals
gbmGrid <-  expand.grid(interaction.depth = c(1,2,3,4,5),
                        n.trees = (1:20)*15,
                        shrinkage = c(0.001,0.01),
                        n.minobsinnode = c(10,15,20))

nrow(gbmGrid)
set.seed(057)
gbmFit1 <- train(trny~trnx, data=data.frame(cbind(trny,trnx)),
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid)
gbmFit1

trellis.par.set(caretTheme())
plot(gbmFit1)
plot(gbmFit1, metric = "Rsquared")

##########################
#test error
#using n.trees = 300, interaction.depth = 5, shrinkage = 0.01 and n.minobsinnode = 10 as the gbm model
gbm.tsterr<-function(trn,tstx,tsty){
model=gbm.fit(trn$x,trn$y,distribution="gaussian",n.trees = 300,shrinkage=0.01,interaction.depth=5,n.minobsinnode = 10)
pred=predict.gbm(model,data.frame(tstx),n.trees=300)
gbm.tsterr=mse(pred,tsty)
return(gbm.tsterr)
}
#unscaled data
gbm.tsterr(trn,tstx,tsty)


### Part 2: random Forest
## Method 1: using 70/30 split cv to tune the parameter mtry
#mtry is the number of independent variables used to include in a tree construction. 
#That's the main difference between random forest and bagging. 
#Bagging uses all independent variables to construct trees.

# random split according to training ratio


tuning.rf<- function(trn,cvec,iters,ratio) {
  set.seed(057)
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      model=randomForest(train$x,train$y,importance=TRUE,mtry = cvec[j])
      pred=predict(model,validation$x,type="response")
      err[i,j]= mean((pred-validation$y)^2)
    }
  }
  
  for (j in 1:nc) {
    print(paste("mtry=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}
##############
mtry=seq(2,20,by=2)
set.seed(057)
tuning.rf(trn,mtry,20,0.7)
#normalized data
tuning.rf(ntrn,mtry,20,0.7)


## Method 2: using caret package
library(caret)

fitControl <- trainControl(
  method = "LGOCV",
  p = 0.7)
# Create the tuning parameter intervals
rfGrid <-  expand.grid(mtry = seq(2,20,by=2))


nrow(rfGrid)
set.seed(057)
rfFit1 <- train(trny~trnx, data=data.frame(cbind(trny,trnx)),
                method = "rf",
                trControl = fitControl,
                tuneGrid = rfGrid)
rfFit1

trellis.par.set(caretTheme())
plot(rfFit1)
plot(rfFit1, metric = "Rsquared")

## train on scaled data
rfFit2 <- train(ntrny~ntrnx, data=data.frame(cbind(ntrny,ntrnx)),
                method = "rf",
                trControl = fitControl,
                tuneGrid = rfGrid)
rfFit2

trellis.par.set(caretTheme())
plot(rfFit2)
plot(rfFit2, metric = "Rsquared")

#test error & featuers importance
rf.tsterr<-function(trn,cvec,tstx,tsty){
  model=randomForest(trn$x,trn$y,importance=TRUE,mtry=cvec)
  pred=predict(model,tstx,type="response")
  rf.tsterr=mse(pred,tsty)
  return(rf.tsterr)
}
#unscaled data
rf.tsterr(trn,6,tstx,tsty)
#scaled data
rf.tsterr(ntrn,8,ntstx,ntsty)

### Part 3:neural networks 
## Method 1: using 70/30 split cv to tune the parameter
tuning.nn<- function(data,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
for (i in 1:iters) {
    cv<-partition.cv(data,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      n<-names(data.frame(train))
      f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
      model<-neuralnet(f,data=data.frame(train),hidden=cvec[j],err.fct="sse",stepmax = 1000000,rep=4,linear.output=TRUE)
      pred<-compute(model,as.matrix(data.frame(validation$x)))$net.result
      err[i,j]= mean((pred-validation$y)^2)
    }
  }
  
  for (j in 1:nc) {
    print(paste("depth=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}
neuron=c(1:10)
set.seed(057)
tuning.nn(trn,neuron,10,0.7)
#normalized data
tuning.nn(ntrn,neuron,10,0.7)

##########################
#test error

nn.tsterr<-function(trn,tstx,tsty,cvec){
  n<-names(data.frame(trn))
  f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
  model<-neuralnet(f,data=data.frame(trn),hidden=cvec,err.fct="sse",linear.output=TRUE)
  pred<-compute(model,as.matrix(data.frame(tstx)))$net.result
  nn.tsterr=mse(pred,tsty)
  return(nn.tsterr)
}
#scaled data
nn.tsterr(trn,tstx,tsty,9)

### Part 4:svm for classification
####tuning svm
library('e1071')
tuning <- function(trn,cvec,iters,ratio,ktype,param) {
  print("=== cross validation error estimation ===")
  nc=length(cvec)
  err=matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv=partition.cv(trn,ratio)
    train=cv$train
    validation=cv$validation
    for (j in 1:nc) {
      if (ktype == 'polynomial') {
        model<-svm(train$x,train$y,kernel='polynomial',type="eps-regression",degree=param,cost=cvec[j])
      }else if (ktype == 'radial'){
        model<-svm(train$x,train$y,kernel="radial",type="eps-regression",gamma=1/(2*param^2),cost=cvec[j])
      }else{
        model=svm(train$x,train$y,kernel="linear",type="eps-regression", cost=cvec[j])
      }
      pred=predict(model,validation$x)
      err[i,j]= err[i,j]= mean((pred-validation$y)^2)
    }
  }
  
  for (j in 1:nc) {
    print(paste("C=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  print(paste( "parameter=",param,"the best cost=",cvec[opt.j], "the error =",merr[opt.j]))
  return(cvec[opt.j])
}


tuning.svm<-function(filename,data,cvec,iters,ratio){
  print(paste(filename))
  print(paste("#########linear SVM#################"))
  tuning(data,cvec,iters,ratio,"linear",1)
  # polynomial kernel
  print(paste("########polynomial kernel SVM##########"))
  params=c(2:7)
  for (deg in params) {
    tuning(data,cvec,iters,ratio,'polynomial',deg)
  }
  # RBF kernel
  print(paste("########RBF kernel SVM###############"))
  params=c(0.05,0.1,0.2,0.3,0.4,0.5,0.8,5)
  for (sigma in params) {
    tuning(data,cvec,iters,ratio,"radial",sigma)
  }}


cvec=c(100,32,10,3.2,1,0.32,0.1,0.032,0.01,0.0032,0.001,0.00032,0.0001)
tuning.svm("original data",trn,cvec,10,0.7)
tuning.svm("normalized data",ntrn,cvec,10,0.7)


#####test error
svm.tsterr<- function(ntrn,ntstx,ntsty,ktype,param,cvec,dataname) {
  print("=== test error ===")
  if (ktype == 'polynomial') {
    model<-svm(ntrn$x,ntrn$y,kernel='polynomial',type="eps-regression",degree=param,cost=cvec,scale = F)
  }else if (ktype == 'radial'){
    model<-svm(ntrn$x,ntrn$y,kernel="radial",type="eps-regression",gamma=1/(2*param^2),cost=cvec,scale = F)
  }else{
    model=svm(ntrn$x,ntrn$y,kernel="linear",type="eps-regression", cost=cvec,scale = F)
  }
  pred=predict(model,ntstx)
  err= mean((pred-ntsty)^2)
  print(paste(dataname,ktype," with parameter =",param,": test error=",err))
}

svm.tsterr(ntrn,ntstx,ntsty,"linear",1,3.2,"normalized data")


### Part 5:linear ridge regression
tuning.rg<- function(trn,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      ww<-myridge.fit(train$x,train$y,lambda[j])
      # compute validation error
      err[i,j]<-mse.linear(ww,validation$x,validation$y)
    }
  }
  
  for (j in 1:nc) {
    print(paste("lambda=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}
lambda=c(1e-4,1e-3,1e-2,0.1,1,10,1e2, 1e3,1e4,1e5,1e6,1e7,1e8)
#with whole dataset
set.seed(0.57)
tuning.rg(ntrn,lambda,10,0.7)
#with seleced variables[ 6 11 8 10 13 ]
v=c(6,8,10,11,13)
new.trn=list(x=ntrnx[,v],y=ntrny)
new.tstx=ntstx[,v]
tuning.rg(new.trn,lambda,10,0.7)

###test error
rg.tsterr<-function(trn,tstx,tsty,lambda){
  ww<-myridge.fit(trn$x,trn$y,lambda)
  err<-mse.linear(ww,tstx,tsty)
  return(err)
}
#scaled data
rg.tsterr(ntrn,ntstx,ntsty,0.001)
#with seleced variables[ 6 11 8 10 13 ]
rg.tsterr(new.trn,new.tstx,ntsty,0.01)


### Part 6:logistic regression
#transform the data
theta=median(trny)
trny.binary=ifelse(trny>=theta,1,0)
tsty.binary=ifelse(tsty>=theta,1,0)
trn.b=list(x=trnx,y=trny.binary)
tst.b=list(x=tstx,y=tsty.binary)
#fit to logistic regression
model=glm.fit(cbind(1,trn.b$x),trn.b$y,family=binomial("logit"))
p=dim(tst.b$x)[2]
pred=tst.b$x%*%model$coef[1:p]+model$coef[p+1]
pred=ifelse(pred>0.5,1,0)
err=sum(pred!=tst.b$y) /length(tst.b$y)
err
###Part 7: model averaging
##using ridge regression to do model averageing


n<-names(data.frame(ntrn$x))
f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
global.model<-glm(f,data=data.frame(ntrn$x,y=ntrn$y),family = gaussian)                     
avg.model <- glmulti(global.model, # use the model with built as a starting point
                     level = 1,  #  just look at main effects
                     crit="aicc") # use AICc because it works better than AIC for small sample sizes
summary(avg.model)
weightable(avg.model)

#using scaled data
f<- glm( y ~ 1 + CRIM + NOX + RM + AGE + DIS + TAX + PTRATIO + B + LSTAT,data=data.frame(ntrn$x,y=ntrn$y),family = gaussian) 
f2 <- glm( y ~ 1 + CRIM + ZN + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT,data=data.frame(ntrn$x,y=ntrn$y),family = gaussian) 
f3 <- glm( y ~ 1 + CRIM + ZN + NOX + RM + AGE + DIS + TAX + PTRATIO + B + LSTAT,data=data.frame(ntrn$x,y=ntrn$y),family = gaussian) 
f4 <- glm( y ~ 1 + CRIM + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT,data=data.frame(ntrn$x,y=ntrn$y),family = gaussian) 


f.ave <- model.avg(f, f2, f3, f4)
summary(f.ave)
pred=predict(f.ave,data.frame(ntstx))
err=mean((pred-ntsty)^2)
err



      #######################################
      ### HW 1.2 Using classification data ##
      #######################################
##  import data
## dataset1:wdbc
read.wbc <- function(file) {
  fdata=read.csv(file,header = FALSE)
  p=dim(fdata)[2]
  x=data.matrix(fdata[,3:p])
  y=data.matrix(fdata[,2]=='M')
  obj=list(x=x,y=y)
  return(obj)
}
wbc.trn=read.wbc('wdbc.train')
wbc.tst=read.wbc('wdbc.test')

## dataset2:hypothyroid
#know about the data set
hyp=read.csv('hypothyroid.train',header = FALSE,na.strings=c("?"),stringsAsFactors=TRUE)
sapply(hyp,function(x) sum(is.na(x)))
sapply(hyp, function(x) length(unique(x)))
library(Amelia)
missmap(hyp, main = "Missing values vs observed")

read.hyp <- function(file) {
  fdata=read.csv(file,header = FALSE,na.strings=c("?"))
  fdata$V2[is.na(fdata$V2)] <- mean(fdata$V2,na.rm=T)
  fdata$V16[is.na(fdata$V16)] <- mean(fdata$V16,na.rm=T)
  fdata$V18[is.na(fdata$V18)] <- mean(fdata$V18,na.rm=T)
  fdata$V20[is.na(fdata$V20)] <- mean(fdata$V20,na.rm=T)
  fdata$V22[is.na(fdata$V22)] <- mean(fdata$V22,na.rm=T)
  fdata$V24[is.na(fdata$V24)] <- mean(fdata$V24,na.rm=T)
  fdata$V26[is.na(fdata$V26)] <- mean(fdata$V26,na.rm=T)
  fdata<-na.omit(fdata)
  p=dim(fdata)[2]
  x=data.matrix(fdata[,2:p])
  y=data.matrix(fdata[1]=='hypothyroid')
  x[,c(2:14,16,18,20,22,24)]=x[,c(2:14,16,18,20,22,24)]-1
  obj=list(x=x,y=y)
  return(obj)
}
hyp.trn=read.hyp('hypothyroid.train')
hyp.tst=read.hyp('hypothyroid.test')



## dataset3:ionosphere
#"Good" radar returns are those showing evidence of some type of structure 
#in the ionosphere.  "Bad" returns are those that do not; their signals pass
#through the ionosphere.  
read.ion <- function(file) {
  fdata=read.csv(file,header = FALSE)
  x=data.matrix(fdata[,3:34])
  y=data.matrix(fdata[,35]=='g')
  obj=list(x=x,y=y)
  return(obj)
}

ion.trn=read.ion('ionosphere.train')
ion.tst=read.ion('ionosphere.test')

###############

### Part 1:gbm 
## Method 1: using 70/30 split cv to tune the parameter
# random split according to training ratio
partition.ind <- function(n, ratio)
{
  train.indices= which(runif(n) < ratio)
  return (train.indices)
}

# partition data into training/validation sets
partition.cv <- function(dat, ratio)
{
  n=length(dat$y)
  trn.ind= partition.ind(n,ratio)
  trn=list(x=dat$x[trn.ind,],y=dat$y[trn.ind])
  val=list(x=dat$x[-trn.ind,],y=dat$y[-trn.ind])
  cv=list(train=trn,validation=val)
  return(cv)
}
# Tuning Parameter using cv

tuningc.gbm<- function(trn,ntrees,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      model=gbm.fit(train$x,train$y,distribution="bernoulli",n.trees = ntrees,shrinkage=0.01,interaction.depth=cvec[j],n.minobsinnode = 10,verbose = FALSE)
      pred=predict.gbm(model,data.frame(validation$x),n.trees=ntrees,type="response")
      pred=ifelse(pred>0.5,1,0)
      err[i,j]= sum(pred!=validation$y) /length(validation$y)
    }
  }
  
  for (j in 1:nc) {
    print(paste("depth=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}
depth=c(1:10)

tuningc.gbm(wbc.trn,300,depth,20,0.7)
tuningc.gbm(ion.trn,450,depth,20,0.7)
tuningc.gbm(hyp.trn,300,depth,10,0.7)
##########################
#test error
tsterr<- function(trn,tst, ntrees,depth,n,dataname) {
  print("=== test error ===")
  model=gbm.fit(trn$x,trn$y,distribution="bernoulli",n.trees = ntrees,shrinkage=0.01,interaction.depth=depth,n.minobsinnode = n,verbose = FALSE)
  pred=predict.gbm(model,data.frame(tst$x),n.trees=ntrees,type="response")
  pred=ifelse(pred>0.5,1,0)
  err= sum(pred!=tst$y) /length(tst$y)
  print(paste(dataname," : test error=",err))
}

wbc.tr<-tsterr(wbc.trn,wbc.tst,500,9,10,"wdbc")
ion.tr<-tsterr(ion.trn,ion.tst,450,9,15,"ionosphere")
hyp.tr<-tsterr(hyp.trn,hyp.tst,300,4,10,"hypothyroid")

#
## Method 2: using caret package
#library(caret)
#fitControl <- trainControl(
#  method = "LGOCV",
#  p = 0.7,classProbs = TRUE,
#  summaryFunction = twoClassSummary)
# Create the tuning parameter intervals
#gbmGrid <-  expand.grid(interaction.depth = c(1:15),
#                        n.trees = seq(50,500,by=50),
#                        shrinkage = 0.01,
#                        n.minobsinnode = c(10,15,20))

# wbc.trn
#set.seed(057)
#wbc.trn$y[wbc.trn$y=="TRUE"] <- "yes"
#wbc.trn$y[wbc.trn$y=="FALSE"] <- "no"

#gbmFit.wbc <- train(y ~ ., data = data.frame(wbc.trn),
#                    method = "gbm",
#                    trControl = fitControl,
#                    verbose = FALSE,
#                    tuneGrid = gbmGrid,metric = "ROC")
#gbmFit.wbc

#trellis.par.set(caretTheme())
#plot(gbmFit.wbc,metric = "ROC")


## ion.trn
#ion.trn$y[ion.trn$y=="TRUE"] <- "yes"
#ion.trn$y[ion.trn$y=="FALSE"] <- "no"

#gbmFit.ion<- train(y~., data=data.frame(ion.trn),
#                 method = "gbm",
#                 trControl = fitControl,
#                 verbose = FALSE,
#                 tuneGrid = gbmGrid,metric = "ROC")
#gbmFit.ion

#trellis.par.set(caretTheme())
#plot(gbmFit.ion)

## hyp.trn
#hyp.trn$y[hyp.trn$y==1] <- "yes"
#hyp.trn$y[hyp.trn$y==0] <- "no"
#n<-names(data.frame(hyp.trn$x))
#f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
#gbmFit.hyp<- train(y~., data=data.frame(x=hyp.trn$x,y=hyp.trn$y),
#                   method = "gbm",
#                   trControl = fitControl,
#                   verbose = FALSE,
#                   tuneGrid = gbmGrid,metric = "ROC")
#gbmFit.hyp

#trellis.par.set(caretTheme())
#plot(gbmFit.hyp)

### Part 2: random Forest
## Method 1: using 70/30 split cv to tune the parameter mtry
#mtry is the number of independent variables used to include in a tree construction. 
#That's the main difference between random forest and bagging. 
#Bagging uses all independent variables to construct trees.

tuningc.rf<- function(trn,cvec,iters,ratio) {
  set.seed(057)
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      model=randomForest(train$x,as.factor(train$y),importance=TRUE,mtry = cvec[j])
      pred=predict(model,validation$x,type="response")
      err[i,j]= sum(pred!=validation$y) /length(validation$y)
    }
  }
  
  for (j in 1:nc) {
    print(paste("depth=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}

mtry=seq(2,20,by=2)
tuningc.rf(wbc.trn,mtry,20,0.7)
tuningc.rf(ion.trn,mtry,20,0.7)
tuningc.rf(hyp.trn,mtry,20,0.7)

##########################
#test error
tsterr<- function(trn,tst,cvec,dataname) {
  print("=== test error ===")
  model=randomForest(trn$x,as.factor(trn$y),importance=TRUE,mtry = cvec)
  pred=predict(model,tst$x,type="response")
  err= sum(pred!=tst$y) /length(tst$y)
  print(paste(dataname," : test error=",err))
}

wbc.tr<-tsterr(wbc.trn,wbc.tst,6,"wdbc")
ion.tr<-tsterr(ion.trn,ion.tst,16,"ionosphere")
hyp.tr<-tsterr(hyp.trn,hyp.tst,18,"hypothyroid")

#
## Method 2: using caret package
#library(caret)
#fitControl <- trainControl(
#  method = "LGOCV",
#  p = 0.7,classProbs = TRUE,
#  summaryFunction = twoClassSummary)
# Create the tuning parameter intervals
#rfGrid <-  expand.grid(mtry = seq(2,20,by=2))

# wbc.trn
#set.seed(057)
#wbc.trn$y[wbc.trn$y=="TRUE"] <- "yes"
#wbc.trn$y[wbc.trn$y=="FALSE"] <- "no"

#rfFit.wbc <- train(y ~ ., data = data.frame(wbc.trn),
#                   method = "rf",
#                 trControl = fitControl,
#                   tuneGrid = rfGrid,metric = "ROC")
#rfFit.wbc

#trellis.par.set(caretTheme())
#plot(rfFit.wbc,metric = "ROC")


## ion.trn
#ion.trn$y[ion.trn$y=="TRUE"] <- "yes"
#ion.trn$y[ion.trn$y=="FALSE"] <- "no"

#rfFit.ion<- train(y~., data=data.frame(ion.trn),method = "rf",trControl = fitControl,tuneGrid = rfGrid,metric = "ROC")
#rfFit.ion

#trellis.par.set(caretTheme())
#plot(rfFit.ion)

## hyp.trn
#hyp.trn$y=as.factor(hyp.trn$y)
#rfFit.hyp<- train(y~., data=data.frame(hyp.trn),method = "rf",trControl = fitControl,tuneGrid = rfGrid,metric = "ROC")
#rfFit.hyp

#trellis.par.set(caretTheme())
#plot(rfFit.hyp)


### Part 3:neural networks 
## Method 1: using 70/30 split cv to tune the parameter

tuningc.nn<- function(trn,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    
    for (j in 1:nc) {
      n<-names(data.frame(train$x))
      f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
      model<-neuralnet(f,data=data.frame(train$x,y=train$y),hidden=cvec[j],err.fct="ce",linear.output=FALSE,stepmax=10000000)
      pred<-compute(model,data.frame(validation$x))$net.result
      pred=ifelse(pred>0.5,1,0)
      err[i,j]= sum(pred!=validation$y) /length(validation$y)
    }
  }
  
  for (j in 1:nc) {
    print(paste("num. of neuron=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}


neuron=c(1:10)
tuningc.nn(wbc.trn,neuron,10,0.7)
tuningc.nn(ion.trn,neuron,10,0.7)
tuningc.nn(hyp.trn,neuron,10,0.7)
##########################
#test error
tsterr<- function(trn,tst,cvec,dataname) {
  print("=== test error ===")
  n<-names(data.frame(trn$x))
  f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
  data=data.frame(trn$y,trn$x)
  colnames(data)[1]="y"
  model<-neuralnet(f,data=data,hidden=cvec,err.fct="ce",linear.output=FALSE,stepmax=10000000)
  pred<-compute(model,data.frame(tst$x))$net.result
  pred=ifelse(pred>0.5,1,0)
  err= sum(pred!=tst$y) /length(tst$y)
  print(paste(dataname," : test error=",err))
}

wbc.tr<-tsterr(wbc.trn,wbc.tst,10,"wdbc")
ion.tr<-tsterr(ion.trn,ion.tst,5,"ionosphere")
hyp.tr<-tsterr(hyp.trn,hyp.tst,3,"hypothyroid")

### Part 4:svm for classification
####tuning svm
library('e1071')
tuning.svm <- function(trn,cvec,iters,ratio,ktype,param) {
  nc=length(cvec)
  err=matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv=partition.cv(trn,ratio)
    train=cv$train
    validation=cv$validation
    for (j in 1:nc) {
      if (ktype == 'polynomial') {
        model<-svm(train$x,train$y,kernel='polynomial',type="C-classification",degree=param,cost=cvec[j])
      }else if (ktype == 'radial'){
        model<-svm(train$x,train$y,kernel="radial",type="C-classification",gamma=1/(2*param^2),cost=cvec[j])
      }else{
        model=svm(train$x,train$y,kernel="linear",type="C-classification", cost=cvec[j])
      }
      pred=predict(model,validation$x)
      err[i,j]= sum(pred != validation$y) /length(validation$y)
    }
  }
  
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  print(paste( "parameter=",param,"the best cost=",cvec[opt.j], "the error =",merr[opt.j]))
  return(cvec[opt.j])
}


tuningc.svm<-function(filename,data,cvec,iters,ratio){
  print(paste(filename))
  print(paste("#########linear SVM#################"))
  tuning.svm(data,cvec,iters,ratio,"linear",1)
  # polynomial kernel
  print(paste("########polynomial kernel SVM##########"))
  params=c(1:7)
  for (deg in params) {
    tuning.svm (data,cvec,iters,ratio,'polynomial',deg)
  }
  # RBF kernel
  print(paste("########RBF kernel SVM###############"))
  params=c(0.05,0.1,0.2,0.3,0.4,0.5,0.8,5)
  for (sigma in params) {
    tuning.svm (data,cvec,iters,ratio,"radial",sigma)
  }}


cvec=c(100,32,10,3.2,1,0.32,0.1,0.032,0.01,0.0032,0.001,0.00032,0.0001)
tuningc.svm("wdbc",wbc.trn,cvec,10,0.7)
tuningc.svm("ionosphere",ion.trn,cvec,10,0.7)
tuningc.svm("hypothyroid",hyp.trn,cvec,10,0.7)

#####test error
tsterr<- function(trn,tst,ktype,param,cvec,dataname) {
  print("=== test error ===")
  if (ktype == 'polynomial') {
    model<-svm(trn$x,trn$y,kernel='polynomial',type="C-classification",degree=param,cost=cvec)
  }else if (ktype == 'radial'){
    model<-svm(trn$x,trn$y,kernel="radial",type="C-classification",gamma=1/(2*param^2),cost=cvec)
  }else{
    model=svm(trn$x,trn$y,kernel="linear",type="C-classification", cost=cvec)
  }
  pred=predict(model,tst$x)
  err= sum(pred!= tst$y) /length(tst$y)
  print(paste(dataname,ktype," with parameter =",param,": test error=",err))
}

#wbc
wbc.tr<-tsterr(wbc.trn,wbc.tst,"linear",1,0.32,"wdbc")
#ion
ion.tr<-tsterr(ion.trn,ion.tst,'radial',5,1,"ionosphere")
#hyp
hyp.tr<-tsterr(hyp.trn,hyp.tst,"linear",1,32,"hypothyroid")


### Part 5:linear ridge regression
tuningc.rg<- function(trn,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc<-length(cvec)
  err<-matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv<-partition.cv(trn,ratio)
    train<-cv$train
    validation<-cv$validation
    for (j in 1:nc) {
      ww<-myridge.fit(train$x,train$y,lambda[j])
      pred=predict.linear(ww,data.frame(validation$x))
      # compute validation error
      pred=ifelse(pred>0,1,0)
      err[i,j]= sum(pred!=validation$y) /length(validation$y)
    }
  }
  
  for (j in 1:nc) {
    print(paste("lambda=",cvec[j]," : error=",mean(err[,j]),"+-",sd(err[,j])))
  }
  merr= colMeans(err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}


lambda=c(1e-4,1e-3,1e-2,0.1,1,10,1e2, 1e3,1e4,1e5,1e6,1e7,1e8)
tuningc.rg(wbc.trn,lambda,10,0.7)
tuningc.rg(ion.trn,lambda,10,0.7)
tuningc.rg(hyp.trn,lambda,10,0.7)

###test error
tsterr<-function(trn,tst,lambda){
  ww<-myridge.fit(trn$x,trn$y,lambda)
  pred=predict.linear(ww,data.frame(tst$x))
  pred=ifelse(pred>0,1,0)
  err<-sum(pred!=tst$y) /length(tst$y)
  return(err)
}

tsterr(wbc.trn,wbc.tst,0.0001)
tsterr(ion.trn,ion.tst,0.0001)
tsterr(hyp.trn,hyp.tst,100000)

### Part 6:logistic regression
#fit to logistic regression
logistic<-function(trn,tst,dataname){
  model=glm.fit(trn$x,trn$y, family = binomial())
  pred=(tst$x)%*%as.matrix(model$coef)
  pred=ifelse(pred>0.5,1,0)
  err=sum(pred!=tst$y) /length(tst$y)
  print(paste(dataname," wirh logistic regression",": test error=",err))
}

logistic(wbc.trn,wbc.tst,"wdbc")
logistic(ion.trn,ion.tst,"ionosphere")
logistic(hyp.trn,hyp.tst,"hypothyroid")

# using logistic regression to do model averageing
library(glmulti)
library(rJava)
model.select<-function(trn,tst){
  train=trn
  test=tst
  n<-names(data.frame(train$x))
  f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
  global.model<-glm(f,data=data.frame(train$x,y=train$y),family = binomial("logit"))
  
  avg.model <- glmulti(global.model, # use the model with built as a starting point
                       level = 1,  #  just look at main effects
                       crit="aicc") # use AICc because it works better than AIC for small sample sizes
  summary(avg.model)
  weightable(avg.model)
}

model.select(wbc.trn,wbc.tst)
model.select(ion.trn,ion.tst)
model.select(hyp.trn,hyp.tst)

############Part 7: model averaging
library(MASS)
library(bbmle)
library(MuMIn)
#  1.for wdbc
train=data.frame(wbc.trn$x,y=wbc.trn$y)
test=wbc.tst
n<-names(data.frame(wbc.trn$x))
f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
model<-glm(f,data=train,family = binomial("logit"))
step <- stepAIC(model, direction="both")
#choose three best model
m1 <- glm(y ~ V4 + V8 + V9 + V15 + V22 + V23 + V29 + V30 + V32, family = binomial("logit"), data = train)
m2 <- glm(y ~ V3 + V4 + V8 + V9 + V15 + V22 + V23 + V29 + V30 + V32, family = binomial("logit"), 
          data = train)
m3 <- glm(y ~ V3 + V4 + V7 + V8 + V9 + V15 + V22 + V23 + V29 + V30 + V32, family = binomial("logit"), 
          data = train)
AICctab(m1, m2, m3, base = T, weights = T, nobs = length(train))
m.ave <- model.avg(m1, m2,m3)
summary(m.ave)
pred=predict(m.ave,data.frame(test$x),backtransform = TRUE)
pred=ifelse(pred>0.5,1,0)
err=sum(pred!=test$y) /length(test$y)
err
#  2.for ion
train=data.frame(ion.trn$x,y=ion.trn$y)
test=ion.tst
n<-names(data.frame(ion.trn$x))
f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
model<-glm(f,data=train,family = binomial("logit"))
step <- stepAIC(model, direction="both")
#choose three best model
m1 <- glm(y ~ V3 + V4 + V5 + V6 + V8 + V9 + V10 + V11 + V13 + V14 + V15 + 
            V16 + V18 + V22 + V23 + V26 + V27 + V30 + V31, family = binomial("logit"), data = train)
m2 <- glm(y ~ V3 + V4 + V5 + V6 + V8 + V9 + V10 + V11 + V13 + V14 + V15 + 
            V16 + V18 + V22 + V23 + V24 + V26 + V27 + V28 + V30 + V31, family = binomial("logit"), 
          data = train)
m3 <- glm(y ~ V3 + V4 + V5 + V6 + V8 + V9 + V10 + V11 + V13 + V14 + V15 + 
            V16 + V18 + V22 + V23 + V24 + V26 + V27 + V30 + V31, family = binomial("logit"), 
          data = train)
AICctab(m1, m2, m3, base = T, weights = T, nobs = length(train))
m.ave <- model.avg(m1, m2,m3)
summary(m.ave)
pred=predict(m.ave,data.frame(test$x),backtransform = TRUE)
pred=ifelse(pred>0.5,1,0)
err=sum(pred!=test$y) /length(test$y)
err
#  3.for hyp
train=data.frame(hyp.trn$x,y=hyp.trn$y)
test=hyp.tst
n<-names(data.frame(hyp.trn$x))
f<-as.formula(paste("V1 ~", paste(n[!n %in% "y"],collapse = "+")))
model<-glm(f,data=train,family = binomial("logit"))
step <- stepAIC(model, direction="both")
#choose three best model
m1 <- glm(V1 ~ V4 + V6 + V14 + V15 + V16 + V24, family = binomial("logit"), data = train)
m2 <- glm(V1 ~V4 + V6 + V7 + V14 + V15 + V16 + V24, family = binomial("logit"), 
          data = train)
m3 <- glm(V1 ~ V4 + V6 + V7 + V14 + V15 + V16 + V17 + V24, family = binomial("logit"), 
          data = train)
AICctab(m1, m2, m3, base = T, weights = T, nobs = length(train))
m.ave <- model.avg(m1, m2,m3)
summary(m.ave)
pred=predict(m.ave,data.frame(test$x),backtransform = TRUE)
pred=ifelse(pred>0.5,1,0)
err=sum(pred!=test$y) /length(test$y)
err

###### Part 8:ROC plot for all the model
# compute fpr tpr for roc plots
roc.points <- function(true.y,predict.score)
{
  ind=sort(as.vector(predict.score),decreasing=FALSE,index.return=TRUE)$ix
  sorted.y=true.y[ind]
  n=length(sorted.y)
  predict.score=predict.score[ind]
  pos=sum(sorted.y>0)
  neg=n-pos
  tpr=rep(0,n-1)
  fpr=rep(0,n-1)
  for(i in 1:(n-1)){
    theta=predict.score[i]
    score=(predict.score>theta)*1
    tpr[i]= sum((score==1)*(sorted.y==TRUE))/pos
    fpr[i]= sum((score==1)*(sorted.y==FALSE))/neg  
  }
  roc.pts=list(tpr=tpr,fpr=fpr)
  return(roc.pts)
}


##gbm model
gbm<- function(trn,tst, ntrees,depth,n,dataname) {
  print("=== test error ===")
  model=gbm.fit(trn$x,trn$y,distribution="bernoulli",n.trees = ntrees,shrinkage=0.01,interaction.depth=depth,n.minobsinnode = n)
  pred=predict.gbm(model,data.frame(tst$x),n.trees=ntrees,type="response")
  return(pred)
}
wbc.gbm<-gbm(wbc.trn,wbc.tst,500,9,10,"wdbc")
ion.gbm<-gbm(ion.trn,ion.tst,450,9,15,"ionosphere")
hyp.gbm<-gbm(hyp.trn,hyp.tst,300,4,10,"hypothyroid")
#rf model
rf<- function(trn,tst,cvec,dataname) {
  model=randomForest(hyp.trn$x,as.factor(hyp.trn$y),importance=TRUE,mtry = cvec)
  pred=predict(model,hyp.tst$x,type="prob")[,"TRUE"]
}

wbc.rf<-rf(wbc.trn,wbc.tst,6,"wdbc")
ion.rf<-rf(ion.trn,ion.tst,16,"ionosphere")
hyp.rf<-rf(hyp.trn,hyp.tst,18,"hypothyroid")

#nn model
nn<- function(trn,tst,cvec,dataname) {
  n<-names(data.frame(trn$x))
  f<-as.formula(paste("y ~", paste(n[!n %in% "y"],collapse = "+")))
  data=data.frame(trn$y,trn$x)
  colnames(data)[1]="y"
  model<-neuralnet(f,data=data,hidden=cvec,err.fct="ce",linear.output=FALSE,stepmax=10000000)
  pred<-compute(model,data.frame(tst$x))$net.result
}

wbc.nn<-nn(wbc.trn,wbc.tst,10,"wdbc")
ion.nn<-nn(ion.trn,ion.tst,5,"ionosphere")
hyp.nn<-nn(hyp.trn,hyp.tst,3,"hypothyroid")

#####SVM model
svmm<- function(trn,tst,ktype,param,cvec) {
  if (ktype == 'polynomial') {
    model<-svm(trn$x,trn$y,kernel='polynomial',type="C-classification",degree=param,cost=cvec,probability = TRUE)
  }else if (ktype == 'radial'){
    model<-svm(trn$x,trn$y,kernel="radial",type="C-classification",gamma=1/(2*param^2),cost=cvec,probability = TRUE)
  }else{
    model<-svm(trn$x,trn$y,kernel="linear",type="C-classification", cost=cvec,probability = TRUE)
  }
  p<-predict(model, data.frame(tst$x),probability = TRUE)
  pred<-attr(p,"probabilities")[,1]
  
}

#wbc
wbc.svm<-svmm(wbc.trn,wbc.tst,"linear",1,0.32)
#ion
ion.svm<-svmm(ion.trn,ion.tst,'radial',5,1)
#hyp
hyp.svm<-svmm(hyp.trn,hyp.tst,"linear",1,32)

###linear ridge regression
rg<-function(trn,tst,lambda){
  ww<-myridge.fit(trn$x,trn$y,lambda)
  pred=predict.linear(ww,data.frame(tst$x))
  
}

wbc.rg<-rg(wbc.trn,wbc.tst,0.0001)
ion.rg<-rg(ion.trn,ion.tst,0.0001)
hyp.rg<-rg(hyp.trn,hyp.tst,100000)

##logistic regression

logistic<-function(trn,tst,dataname){
  model=glm.fit(trn$x,trn$y, family = binomial())
  pred=(tst$x)%*%as.matrix(model$coef)
}

wbc.lg<-logistic(wbc.trn,wbc.tst,"wdbc")
ion.lg<-logistic(ion.trn,ion.tst,"ionosphere")
hyp.lg<-logistic(hyp.trn,hyp.tst,"hypothyroid")

##roc
roc <- function(gbm,rf,nn,svm,rg,lg,fpr.end,tst,dataname) {
  print("=== roc ===")
  pred=gbm
  roc.gbm=roc.points(tst$y,pred)
  
  pred=rf
  roc.rf=roc.points(tst$y,pred)
  
  pred=nn
  roc.nn=roc.points(tst$y,pred)
  
  pred=svm
  roc.svm=roc.points(tst$y,pred)
  
  pred=rg
  roc.rg=roc.points(tst$y,pred)
  
  pred=lg
  roc.lg=roc.points(tst$y,pred)
  
  outfn="roc.pdf";
  cat(paste("output plot to ",outfn,"\n"))
  pdf(file=outfn)
  fpr=roc.gbm$fpr
  tpr=roc.gbm$tpr
  ii=which(fpr<=fpr.end)
  plot(fpr[ii],tpr[ii],xlab="fpr",ylab="tpr", type="p",pch=4,main=(paste("ROC for",dataname)),col=1)
  
  fpr=roc.rf$fpr
  tpr=roc.rf$tpr
  ii=which(fpr<=fpr.end)
  points(fpr[ii],tpr[ii], type="p",pch=5,col=2)
  
  fpr=roc.nn$fpr
  tpr=roc.nn$tpr
  ii=which(fpr<=fpr.end)
  points(fpr[ii],tpr[ii], type="p",pch=6,col=3)
  
  fpr=roc.svm$fpr
  tpr=roc.svm$tpr
  ii=which(fpr<=fpr.end)
  points(fpr[ii],tpr[ii], type="p",pch=7,col=4)
  
  fpr=roc.rg$fpr
  tpr=roc.rg$tpr
  ii=which(fpr<=fpr.end)
  points(fpr[ii],tpr[ii],type="p",pch=8,col=5)
  
  fpr=roc.lg$fpr
  tpr=roc.lg$tpr
  ii=which(fpr<=fpr.end)
  points(fpr[ii],tpr[ii],type="p",pch=9,col=6)
  
  
  legend(x="bottomright",legend=c("gbm","rf","nn","svm","ridge","logistic"), pch=c(4:9),col=c(1:6))
  dev.off()
}
#plot the roc
roc(wbc.gbm,wbc.rf,wbc.nn,wbc.svm,wbc.rg,wbc.lg,0.2,wbc.tst,"wdbc")
roc(ion.gbm,ion.rf,ion.nn,ion.svm,ion.rg,ion.lg,0.5,ion.tst,"ionosphere")
roc(hyp.gbm,hyp.rf,hyp.nn,hyp.svm,hyp.rg,hyp.lg,0.2,hyp.tst,"hypothyroid")

