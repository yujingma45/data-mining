# 1. install lars: starts R, and within R use the following:
#    install.pakcages("lars")
#
# 2. use the following template for your solution
#  (a) read and understand the provided code
#  (b) replace code in places marked with "fill in code"
#

# load lasso 
#
library("lars")

# compute normalization factor
normfact <- function (x) {
  if (is.vector(x)) {
  n<-length(x)
  }
  else{
    n<-nrow(x)
  }
  xsqr<- rep(1/n, n)%*%x^2
  nfact<- drop(xsqr)^0.5
  #another solution: nfact<- drop(t(as.vector(rep(1/n, n))) %*% x^2)^0.5 
  return (nfact)
}

# normalize
normalize <- function (x, nf) {
  n<-nrow(x)
  p<-ncol(x)
  x <- x/rep(nfact, rep(n, p))
# fill in code
  return(x)
}

# centering with respect to mean mu
centering <- function (x, mu) {
  
  if (is.vector(x)) {
    muvector <- rep(mu, length(x))
    x<-x-muvector       
    
  }
  else {
    n<-nrow(trnx)
    p<-ncol(trnx)
    mumatrix<-matrix(mux, nrow = n, ncol = p, byrow = TRUE)
    x<-x-mumatrix
 }
  return(x)
}

# MSE (mean squared error) between predicted-y py and true-y ty
mse <- function(py,ty) {
  return(mean((py-ty)^2))
}

# compute ridge regression weight
myridge.fit <- function(X,y,lambda) {
  w= matrix(1,length(lambda))
  for (i in 1:length(lambda))
  {
    w[1,i]<-solve((X)%*%X+lambda[i]*((X)))%*%(X)%*%y
  }
  # fill in code--ridge regression
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

# select the best s features from path
features.from.path <- function(path, s) {
  k=0;
  bestj=1
  besti=path[1]
  mys=1;
  kk=rep(0,s);
  for (j in 1:length(path)) {

    if (path[j]>0) {
      k=k+1;
      kk[k]=path[j];
    }
    else {
      ik=which(kk[1:k]==-path[j]);
      kk[ik[1]]=kk[k];
      k=k-1;
    }
    if ((mys<k) & (k<=s)) {
      mys=k
    }
    if (k==mys) {
      besti=kk[1:k];
      bestj=j;
    }
  }
  return (besti);
}

# compute variable importance using F-score
feature.Fscore <- function (X,y) {
  p=dim(X)[2]
  score=rep(0,p)
  # fill in code
  
  return(score)
}

# compute variable importance using correlation
feature.Cor <- function (X,y) {
  p=dim(X)[2]
  score=rep(0,p)
  for (i in 1:p){
    score[i]=cor(X[,i],y)
  }
#another way:cor(x,y)
  return(score)
}

# plot training/test error versus lambda and save to outfile
#
myplot <- function(lambda,trnerr,tsterr,outfile) {
  ymin=min(min(trnerr),min(tsterr))
  ymax=max(max(trnerr),max(tsterr))
  
  pdf(file=outfile)
  plot(lambda,trnerr,ylim=c(ymin,ymax),ylab="mean squared error", type="l",log="x",lty=3)
  lines(lambda,tsterr,lty=1)
  legend(x="topleft",legend=c("training error","test error"), lty=c(3,1))
  dev.off()
}

#
# plot training and test error with respect to lambda
# for ridge regression and lasso, and output to pdf file
#
prob1.1 <- function(trnx, trny, tstx, tsty, lambda) {

  # ridge regression
  
  trnerr=lambda
  tsterr=lambda

  # loop through regualrization parameters
  for (i in 1:length(lambda)) {
    # compute coefficient using ridge regression
    ww<-myridge.fit(trnx,trny,lambda[i])
    # compute training error
    trnerr[i]<-mse.linear(ww,trnx,trny)
    # compute test error
    tsterr[i]<-mse.linear(ww,tstx,tsty)
  }

  # plot training/test error for ridge regression
  outfn=paste("prob1.1","-ridge.pdf",sep="")
  cat(paste("output plot to ",outfn,"\n"))
  myplot(lambda,trnerr,tsterr,outfn)
  
  # lasso regression
  
  trnerr2=lambda
  tsterr2=lambda

  # form lasso model
  lasso=0
  # loop through regualrization parameters
  for (i in 1:length(lambda)) {
    # fill in code -- replace the following with coefficient of lasso using coef.lars 
    ww=rep(0,dim(trnx)[2])
    # fill in code --- replace the following with correct code
    trnerr2[i]<-0
    tsterr2[i]<-0
  }

  # plot training/test error for lasso
  outfn=paste("prob1.1","-lasso.pdf",sep="")
  cat(paste("output plot to ",outfn,"\n"))
  myplot(lambda,trnerr2,tsterr2,outfn)
  cat("\n\n")
}


prob1.2 <- function(trnx, trny) {
  cat("features ranked through F-score:\n")
  varimpF= feature.Fscore(trnx,trny)
  path.F <<- sort( varimpF ,decreasing=TRUE,index.return=TRUE)$ix
  print(path.F)
  cat("---\n\n")

  cat("features ranked through correlation:\n")
  varimpCor= feature.Cor(trnx,trny)
  path.Cor <<- sort( varimpCor ,decreasing=TRUE,index.return=TRUE)$ix
  print(path.Cor)
  cat("---\n\n")

  cat("features ranked through Least Squares coefficients:\n")
  varimpLS= abs(myridge.fit(trnx,trny,1e-10))
  path.LS <<- sort( varimpLS ,decreasing=TRUE,index.return=TRUE)$ix
  print(path.LS)
  cat("---\n\n")

  cat("feature ranked through Ridge coefficients (lambda=1):\n")
  varimpRidge= abs(myridge.fit(trnx,trny,1))
  path.Ridge <<- sort( varimpRidge,decreasing=TRUE,index.return=TRUE)$ix
  print(path.Ridge)
  cat("---\n\n")
}

prob1.3 <- function(trnx, trny) {

  # forward feature selection
  # fill in code --- replace the following using lars()
  path.forward = c(1:13)
  cat("features ranked through forward feature selection:\n")
  print(path.forward)
  cat("---\n\n")

}

prob1.4 <- function(trnx,trny) {
  # lasso path
  # fill in code --- replace the following using lars()
  path.lasso = c(1:13)
  cat("lasso path\n")
  print(path.lasso)
  # find top three features
  f3=features.from.path(path.lasso,3)
  cat("top three features\n")
  print(f3)
  cat("---\n\n")

}
  
# evaluate best k features of path on data (X,y)
eval.path <- function(pa,k,trnx,trny,tstx,tsty) {
  # find the best k features
  best.features=features.from.path(pa,k)
  cat("features=[",best.features,"] ",sep=" ")

  
  # least squares fit on training data
  xp=trnx[,best.features];
  if (is.vector(xp)) {
    dim(xp) <- c(length(xp),1)
  }
  ww=myridge.fit(xp,trny,1e-10)

  # mean-squared error on training data
  # cat("train-error=",mse.linear(ww,xp,trny)," ",sep="")

 # mean-squared error on test data
 # fill-in code: replace with mean squared error on test data
 err = 0
  cat("test-error=",err,"\n",sep="")
}

prob1.5 <- function(trnx,trny,tstx,tsty) {
  for (k in c(1,2,3,4,5)) {
    cat("k=",k,"\n",sep="")

    cat(" F-score: ",sep="")
    eval.path(path.F,k,trnx,trny,tstx,tsty)

    cat(" LS-weight: ",sep="")
    eval.path(path.LS,k,trnx,trny,tstx,tsty)

    cat(" ridge-weight: ",sep="")
    eval.path(path.Ridge,k,trnx,trny,tstx,tsty)

    cat(" foward: ",sep="")
    eval.path(path.forward,k,trnx,trny,tstx,tsty)

    cat(" lasso: ",sep="")
    eval.path(path.lasso,k,trnx,trny,tstx,tsty)

  }
}

header<-c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV")

housing<-data.matrix(read.table("housing.data",col.names=header))

trnx<-housing[1:253,1:13]
trny<-housing[1:253,14]

tstx<-housing[254:506,1:13]
tsty<-housing[254:506,14]

#centering
mux=apply(trnx,2,mean)
muy=mean(trny)
trnx=centering(trnx,mux)
trny=centering(trny,muy)
tstx=centering(tstx,mux)
tsty=centering(tsty,muy)

#normalization

nfact=apply(trnx,2,normfact)
trnx=normalize(trnx,nfact)
tstx=normalize(tstx,nfact)


#regularization parameter lambda
lambda=c(1e-4,1e-3,1e-2,0.1,1,10,1e2, 1e3,1e4,1e5,1e6,1e7,1e8)

prob1.1(trnx,trny,tstx,tsty, lambda)
prob1.2(trnx,trny)
prob1.3(trnx,trny)
prob1.4(trnx,trny)
prob1.5(trnx,trny,tstx,tsty)


