                    ###Assignment2 prob1####
                                        # by Yujing Ma
# install and use the MASS library
library("MASS")
# generating isotropic gaussians
# n data for each class 1,2,3 in p-dimension
isogs <- function(n,p) {
  X=matrix(rnorm(3*n*p),nrow=3*n,ncol=p)
  for (i in 1:n) {
    X[i,1]=X[i,1]+3;
    X[n+i,1]=X[n+i,1]-3;
    X[2*n+i,1:2]=X[2*n+i,1:2]+c(1,-1);
  }
  y=c(rep(1,n),rep(2,n),rep(3,n))
  data = list(X=X,y=y)
  return(data)
}

# generate anisotropic gaussians
# n data for each class 1,2,3 in p-dimension
anisogs <- function(n,p) {
  X=matrix(rnorm(3*n*p),nrow=3*n,ncol=p)
  X[,1]=X[,1]*0.5;
  X[,2]=X[,2]*1.5;
  X[,3:p]=X[,3:p]*matrix(runif(3*n*(p-2),min=0,max=3),nrow=3*n,ncol=p-2);
  for (i in 1:n) {
    X[i,1]=X[i,1]+3;
    X[n+i,1]=X[n+i,1]-3;
    X[2*n+i,1:2]=X[2*n+i,1:2]+c(1,-1);
  }
  y=c(rep(1,n),rep(2,n),rep(3,n))
  data = list(X=X,y=y)
  return(data)
}


# scatter plot data to 2-d
# X: data matrix
# y: label from 1-3
# v: v$v1 and v$v2 are two projection directions
# outfn: output file name
scatterplot <- function(X,y,v,outfn) {
  cat(paste("output plot to ",outfn,"\n"))
  Xp=pj(X,v)
  pdf(file=outfn)
  i1=which(y==1,TRUE);
  plot(Xp[i1,],xlim=c(min(Xp[,1]),max(Xp[,1])),ylim=c(min(Xp[,2]),max(Xp[,2])),type="p",pch='1',col='red')
  i2=which(y==2,TRUE);
  points(Xp[i2,],type="p",pch='2',col='green')
  i3=which(y==3,TRUE);
  points(Xp[i3,],type="p",pch='3',col='blue')
  dev.off()
}

# projection of data matrix X to 2-dimension [v$v1,v$v2]
pj <- function(X, v) {
  Xp=matrix(nrow=dim(X)[1],ncol=2)
  Xp[,1]=X%*%matrix(v$v1,ncol=1)
  Xp[,2]=X%*%matrix(v$v2,ncol=1)
  return(Xp)
}

# projection of data$X to 2-dimension [v$v1,v$v2] and align with data$y
pj2 <- function(data,v) {
  Xp=matrix(nrow=length(data$y),ncol=2)
  Xp[,1]=data$X%*%matrix(v$v1,ncol=1)
  Xp[,2]=data$X%*%matrix(v$v2,ncol=1)
  return (list(X=Xp,y=data$y))
}

# generating two orthogonal random-directions
randpj <- function(X) {
  p=dim(X)[2]
  v1=rnorm(p)
  v1=v1/sqrt(sum(v1*v1))
  v2=rnorm(p)
  v2=v2-sum(v1*v2)*v1
  v2=v2/sqrt(sum(v2*v2))
  v=list(v1=v1,v2=v2)
  return(v)
}

# top2-pca-directions, assuming centered X
pcapj <- function(X) {
  # fill in code:
  # use svd() function, and return the first dimension as v$v1
  # and second as v$v2
  # 
       v1=svd(X)$v[,1]
       v2=svd(X)$v[,2] 
       v=list(v1=v1,v2=v2)
  return(v)
}


# top2-lda-directions, assuming centered X
ldapj <- function(data) {
  # fill in code:
  # use lda() function, and return the first dimension as v$v1
  # and second as v$v2
  # 
  fit<<-lda(data$X,data$y)
  first_dimension=fit$scaling[,1]
  second_dimension =fit$scaling[,2]
  v=list(v1= first_dimension,v2= second_dimension)
  return(v)  
}

# one versus all training with generalized linear model
oneversusall.fit <- function(data,fam) {
  # fill in code:
  # use glm.fit() function
  # w1: y=1 versus others 
y1=data$y
x=cbind(1,data$X)
for (i in 1:length(y1)){
       if (y1[i]!=1){
		y1[i]=0
       }
}
w1=glm.fit(x,y1, family = fam,intercept=TRUE)$coef

  # w2: y=2 versus others

y2=data$y
for (i in 1:length(y2)){
	ifelse(y2[i]==2, y2[i]<-1 , y2[i]<-0)

}
w2=glm.fit(x,y2, family = fam, intercept=TRUE)$coef

  # w3: y=3 versus others
  

y3=data$y
for (i in 1:length(y3)){
	ifelse(y3[i]==3, y3[i]<-1 , y3[i]<-0)

}
w3=glm.fit(x,y3, family = fam, intercept=TRUE)$coef

  w=list(w1=w1,w2=w2,w3=w3)
  return(w)
}


# one versus all classification 
predict.oneversusall <- function(w,X) {
X= cbind(1, X)

  n=dim(X)[1]
  p=dim(X)[2]
  y=rep(1,n)
  for (i in 1:n) {
    xx=X[i,]
    # s1: score for class 1
    # s2: score for class 2
    # s3: score for class 3
    s1=xx%*%as.matrix(w$w1)
    s2=xx%*%as.matrix(w$w2)
    s3=xx%*%as.matrix(w$w3)

    # y is the label with maximum score
    if (max(s1,s2,s3)==s1){
	y[i]=1
    }else{
	if (max(s1,s2,s3)==s2){
		y[i]=2
		}else{
		y[i]=3
			}
		}
	}
  return(y)
}

#classification error between predicted label vector py and true label vector ty
# 
cerr <- function(py,ty) {
  err= sum(py != ty)/length(py)
  return(err)
}

#classification error between predicted label vector py and true label vector ty:
# try to assign the optimal label correspondence btween label values for py (clustering output) and ty
#
cerr.bestmatch <- function(py,ty) {
  K=3
  # find best correspondence between label value in py and ty
  # assuming K=3 classes
  for (i in (1:(K-1))) {
    ii=which(ty==i)
    for (j in ((i+1):K)) {
      if (sum(py[ii]==i) <sum(py[ii]==j)) {
        py[which(py==i)]=(K+1)
        py[which(py==j)]=i
        py[which(py==(K+1))]=j
      }
    }
  }
  return (cerr(py,ty))
}

prob1.1 <- function(trn,tst,label) {

  # prob1.1
  #random projection
  vrand <<- randpj(trn$X)

  scatterplot(trn$X,trn$y,vrand,paste("prob1-",label,"-rand.pdf",sep=""))

  # pca projection
  vpca <<- pcapj(trn$X)
  cat("first 5 components of 1st PC\n")
  print(vpca$v1[1:5])
  cat("first 5 components of 2nd PC\n")
  print(vpca$v2[1:5])
  scatterplot(trn$X,trn$y,vpca,paste("prob1-",label,"-pca-trn.pdf",sep=""))
  scatterplot(tst$X,trn$y,vpca,paste("prob1-",label,"-pca-tst.pdf",sep=""))

  # lda projection
  # fill in code
  vlda<<-ldapj(trn)
  cat("first 5 components of 1st LDC\n")
  print(vlda$v1[1:5])
  cat("first 5 components of 2nd LDC\n")
  print(vlda$v2[1:5])
  scatterplot(trn$X,trn$y,vlda,paste("prob1-",label,"-lda-trn.pdf",sep=""))
  scatterplot(tst$X,trn$y,vlda,paste("prob1-",label,"-lda-tst.pdf",sep=""))

}


prob1.2 <- function(trn,tst,label) {
  z<-lda(trn$X,trn$y)
  py= c(rep(1,3*n))
  for (i in 1:length(tst$y)) {
    py[i]= predict(z,tst$X[i,])$class
  }
  err=cerr(py,tst$y)
  cat("lda ",label," error =",err,"\n")
  
}

prob1.34 <- function(trn,tst) {
  # problem 1.3
  # least squares 
  w=oneversusall.fit(trn,gaussian())
  err=cerr(predict.oneversusall(w,trn$X),trn$y)
  cat(paste("  least squares training error =",err,"\n"))
  err=cerr(predict.oneversusall(w,tst$X),tst$y)
  cat(paste("  least squares test error =",err,"\n"))
  
  # logistic regression 
  w=oneversusall.fit(trn,binomial())
  err=cerr(predict.oneversusall(w,trn$X),trn$y)
  cat(paste("  logistic training error =",err,"\n"))
  err=cerr(predict.oneversusall(w,tst$X),tst$y)
  cat(paste("  logistic test error =",err,"\n"))

  # problem 1.4
  # fill in code:
  # K-Means Cluster Analysis, assuming 3 classes
  kmean <- kmeans(tst$X, 3) 
  # use kmeans() to cluster, and return py as cluster label
  py=kmean$cluster
  # find the best label correspondence to true label tst$y
  err=cerr.bestmatch(py,tst$y)
  cat(paste("  kmeans clustering error =",err,"\n"))
}


# do the experiments with train data Xtrn and test data Xtst
# label is the dataset: isotropic or anisotropic
#
doexp <- function(trn,tst,label) {
  mu=colMeans(trn$X)
  trn$X=t(t(trn$X)-mu)
  tst$X=t(t(tst$X)-mu)
  
  cat(paste("---",label,"---\n"))

  prob1.1(trn,tst,label)

  prob1.2(trn,trn,"training")
  prob1.2(trn,tst,"test")

  # without dimension reduction
  cat("without dimension reduction\n")
  prob1.34(trn,tst)

  # random
  cat("random dimension reduction\n")
  prob1.34(pj2(trn,vrand),pj2(tst,vrand))

  # pca
  cat("pca dimension reduction\n")
  prob1.34(pj2(trn,vpca),pj2(tst,vpca))

  # lda
  cat("lda dimension reduction\n")
  prob1.34(pj2(trn,vlda),pj2(tst,vlda))
  cat("\n")

}

p=200
n=100

set.seed(76368)

trn1=isogs(n,p)
tst1=isogs(n,p)

trn2=anisogs(n,p)
tst2=anisogs(n,p)


doexp(trn1,tst1,"isotropic")


doexp(trn2,tst2,"anisotropic")


