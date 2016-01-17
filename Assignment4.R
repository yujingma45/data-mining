# libsvm
#
library('e1071')

# generate labels
prob1.genlabel <- function(x) {
  y=((x[,1] <= -0.5) | (x[,1]^2+x[,2]^2 <=1))*2 -1
  return(y)
}

# generate test data grid
set.seed(057)
prob1.gentstx <- function(n) {
  x=matrix(0,n*n,ncol=2)
  for (i in 1:n) {
    for (j in 1:n) {
      x[(i-1)*n+j,]= c(-2.05 + 4.1*(i-1)/(n-1), -2.05 + 4.1*(j-1)/(n-1)) 
    }
  }
  return(x)
}

# compute decision boundary using test grid
prob1.boundary.find <- function(x,y) {
  nn=length(y)
  n=floor(0.5+sqrt(nn))
  ii=which((y[1:(nn-n)] * y[2:(nn-n+1)] <=0) | (y[1:(nn-n)]*y[(1+n):nn]<=0))
  bx=x[ii,]
  return(bx)
}

# plot decision boundarys
prob1.boundary.plot <- function(boundary.true,boundary.pred,filename) {
  pdf(file=filename)
  plot(boundary.true,xlab="x1",ylab="x2",xlim=c(-2,2),ylim=c(-2,2),type="p",pch='.',col='green')
  points(boundary.pred,type="p",pch='o',col='red')
  legend(x="bottomright",legend=c("true boundary","predicted boundary"), pch=c('.','o'),col=c('green','red'))
  dev.off()
}


# value of kernel ker
# ker$type: 'RBF' or 'polynomial'
# ker$param: kernel parameter (bandwidth or degree)
#
kernel.eval <- function(x1,x2,ker) {
  k=0
  if (ker$type == 'RBF') {
    # RBF kernel
    k=exp(-sum((x1-x2)*(x1-x2)/(2*ker$param^2)))
  }
  else {
    # polynomial kernel
    k=(1+sum(x1*x2))^ker$param
  }
  return(k)
}


# train on trn.x, trn.y using kernel ridge regression with regularization lambda,
# and test on tst.x
kernel.ridge <- function(trn.x, trn.y, tst.x, tst.y, lambda, ker) {
  # kernel ridge regression

  n1<-dim(trn.x)[1]
  n2<-dim(tst.x)[1]
  
  trn.k=matrix(0,n1,n1)
  for(j in 1:n1){
    for (i in 1:n1){
  
      trn.k[i,j]=kernel.eval(trn.x[i,],trn.x[j,],ker)
  
  }}
  
  tst.k=matrix(0,n2,n1)
  for( i in 1:n2){
    for (j in 1:n1){
      
      tst.k[i,j]=kernel.eval(tst.x[i,],trn.x[j,],ker)
      
    }}
  
  
  w=solve((t(trn.k) %*% trn.k) +(lambda*diag(dim(trn.k)[2])), (t(trn.k) %*% trn.y))
  
  tst.pred<-c(array(0,dim=c(n2,1)))
  for (i in 1:n2) {
    tst.pred[i]=sum(w*tst.k[i,])
  }
  
  err= sum(tst.pred*tst.y<=0)/length(tst.pred)
  cat(paste(" kernel ridge regression test error=",err,'\n',sep=""))
  
  # find boundaries
  boundary.true=prob1.boundary.find(tst.x,tst.y)
  boundary.pred=prob1.boundary.find(tst.x,tst.pred)

  # plot true boundary and predicted-boundary
  outfn=paste("prob1-ridge-",ker$type,"-",ker$param,".pdf",sep="")
  prob1.boundary.plot(boundary.true,boundary.pred,outfn)
}

# train on trn.x, trn.y using kernel smoothing
# and test on tst.x
kernel.smoothing <- function(trn.x, trn.y, tst.x, tst.y, ker) {
  # kernel smoothing
  n1<-dim(trn.x)[1]
  n2<-dim(tst.x)[1]
  
  trn.k=matrix(0,n1,n1)
  for(j in 1:n1){
    for (i in 1:n1){
      
      trn.k[i,j]=kernel.eval(trn.x[i,],trn.x[j,],ker)
      
    }}
  
  tst.k=matrix(0,n2,n1)
  for( i in 1:n2){
    for (j in 1:n1){
      
      tst.k[i,j]=kernel.eval(tst.x[i,],trn.x[j,],ker)
      
    }}
  
  tst.pred=c(array(0,dim=c(n2,1)))
  for (i in 1:n2) {
    tst.pred[i]=sum(tst.k[i,]*trn.y)/sum(tst.k[i,])
  }
  

  err<-sum(tst.pred*tst.y<=0)/length(tst.pred)
  cat(paste(" kernel smoothing test error=",err,'\n',sep=""))

  # find boundaries
  boundary.true=prob1.boundary.find(tst.x,tst.y)
  boundary.pred=prob1.boundary.find(tst.x,tst.pred)

  # plot true boundary and predicted-boundary
  outfn=paste("prob1-smoothing-",ker$type,"-",ker$param,".pdf",sep="")
  prob1.boundary.plot(boundary.true,boundary.pred,outfn)
}

# train on trn.x, trn.y using svm
# and test on tst.x
kernel.svm <- function(trn.x,trn.y, tst.x, tst.y, lambda, ker) {
  # kernel svm
  if (ker$type == 'polynomial') {
    model<-svm(trn.x,trn.y,kernel='polynomial',degree=ker$param,coef0=1,cost=1/lambda,gamma=1)
  }else {
    model<-svm(trn.x,trn.y,kernel="radial",gamma=1/(2*ker$param^2),cost=1/lambda)
  }
  tst.pred=predict(model,tst.x)
  
  err=sum(tst.pred*tst.y<=0)/length(tst.pred)
  cat(paste(" kernel svm test error=",err,'\n',sep=""))

  # find boundaries  
  boundary.true=prob1.boundary.find(tst.x,tst.y)
  boundary.pred=prob1.boundary.find(tst.x,tst.pred)

  # plot true boundary and predicted-boundary
  outfn=paste("prob1-svm-",ker$type,"-",ker$param,".pdf",sep="")
  prob1.boundary.plot(boundary.true,boundary.pred,outfn)
}

# generate training data
n=100
trn.x=matrix(runif(n*2,-2,2),ncol=2);
trn.y=prob1.genlabel(trn.x)

# generate test data
tst.x=prob1.gentstx(101)
tst.y=prob1.genlabel(tst.x)


# polynomial kernel

params=c(1:7)
for (deg in params) {
  ker=list(type='polynomial',param=deg)
  cat(paste('kernel=',ker$type,' degree=',ker$param,"\n",sep=""))
  lambda=0.6
  kernel.ridge(trn.x,trn.y,tst.x,tst.y,lambda,ker)
  kernel.svm(trn.x,trn.y,tst.x,tst.y,lambda,ker)
}

# RBF kernel

params=c(0.05,0.1,0.2,0.3,0.4,0.5,0.8,5)
for (sigma in params) {
  ker=list(type='RBF',param=sigma)
  cat(paste('kernel=',ker$type,' sigma=',ker$param,"\n",sep=""))
  lambda=0.6
  kernel.ridge(trn.x,trn.y,tst.x,tst.y,lambda,ker)
  kernel.smoothing(trn.x,trn.y,tst.x,tst.y,ker)
 kernel.svm(trn.x,trn.y,tst.x,tst.y,lambda,ker)
}





