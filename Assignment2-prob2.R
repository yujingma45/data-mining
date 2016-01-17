                     ###Assignment2 prob2####
# compute normalization factor
normfact <- function (x) {
  return(sqrt(mean(x^2)))
}

# normalize
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

# MSE (mean squared error) between predicted-y py and true-y ty
mse <- function(py,ty) {
  return(mean((py-ty)^2))
}

# compute ridge regression weight
myridge.fit <- function(X,y,lambda) {
  w= solve((t(X) %*% X) +(lambda*diag(dim(X)[2])),(t(X) %*% y))
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


header<-c("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO
","B","LSTAT","MEDV")

housing<-data.matrix(read.table("housing.data",col.names=header))


trnx<-housing[1:253,1:13]
trny<-housing[1:253,14]

tstx<-housing[254:506,1:13]
tsty<-housing[254:506,14]

#normalization
nfact=apply(trnx,2,normfact)
trnx=normalize(trnx,nfact)
tstx=normalize(tstx,nfact)


#centering
mux=apply(trnx,2,mean)
muy=mean(trny)
trnx=centering(trnx,mux)
trny=centering(trny,muy)
tstx=centering(tstx,mux)
tsty=centering(tsty,muy)

################## the above are copied from assignment 1 #################

##### problem 2.1
p=dim(trnx)[2]
r=2:p
trn.err=rep(0,p-1)
tst.err=rep(0,p-1)

#compute PCA

for (k in 1:length(r)) { 
   c=k+1
   #compute PCA
   pc<-svd(trnx)$v[,1:c] 
   #compute projection
   pjtrnx<-trnx%*%pc
   pjtstx<-tstx%*%pc

  # fit the model
  ww<-myridge.fit(pjtrnx,trny,0)
 #predict training and test y
  trnpy<- predict.linear(ww,pjtrnx)
  tstpy<- predict.linear(ww,pjtstx)
 # compute training/test error
  trn.err[k]= mse(trnpy,trny)
  tst.err[k]= mse(tstpy,tsty)

  cat(paste("reduced-dimension r=",r[k], ' training-error=', trn.err[k],' test-error=', tst.err[k], '\n'))
}

##### problem 2.2

theta=median(trny)
trny.binary=1+ (trny>=theta)
tsty.binary=1+ (tsty>=theta)

# Compute pca projection 
pc<-svd(trnx)$v 
v1<-pc[,1]
v2<-pc[,2]
v=list(v1=v1,v2=v2)
pjtrnx<-trnx%*%pc[,1:2]
pjtstx<-tstx%*%pc[,1:2]


# scatter plot data to 2-d
# Xp: data matrix projection
# y: label from 1-3
# v: v$v1 and v$v2 are two projection directions
# outfn: output file name
scatterplot <- function(Xp,y,v,outfn) {
  cat(paste("output plot to ",outfn,"\n"))
  pdf(file=outfn)
  i1=which(y==1,TRUE);
  plot(Xp[i1,],xlim=c(min(Xp[,1]),max(Xp[,1])),ylim=c(min(Xp[,2]),max(Xp[,2])),type='p',pch='1',col='red')
  i2=which(y==2,TRUE);
  points(Xp[i2,],type='p',pch='2',col='blue')
  dev.off()
}

 
scatterplot(pjtrnx, trny.binary,v,paste("prob2-pca-trn.pdf",sep=""))
scatterplot(pjtstx, tsty.binary,v,paste("prob2-pca-tst.pdf",sep=""))
