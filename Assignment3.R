###Assignment3 ####
                  # by Yujing Ma



library('e1071')

# reading breast cancer data files
read.data <- function(file) {
  fdata=read.csv(file)
  p=dim(fdata)[2]
  x=data.matrix(fdata[,3:p])
  y=data.matrix(fdata[,2]=='M')
  obj=list(x=x,y=y)
  return(obj)
}

# partition data into training/validation sets
partition.cv <- function(dat, ratio)
{dat1<-data.frame(dat)
smp_size <- floor(ratio * nrow(dat1))
trainindex <- sample(seq_len(nrow(dat1)), size = smp_size)

  trn=list(x=dat$x[trainindex,],y= dat$y[trainindex])
  val=list(x= dat$x[-trainindex,],y= dat$y[-trainindex])
  cv=list(train=trn,validation=val)
  return(cv)
}


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

prob1.1 <- function(trn,cvec,iters,ratio) {
  print("=== cross validation error estimation ===")
  nc=length(cvec)
  validation.err=matrix(rep(0,iters*nc),nrow=iters)
  for (i in 1:iters) {
    cv=partition.cv(trn,ratio)
    train=cv$train
    validation=cv$validation
    for (j in 1:nc) {
       model<-svm(x=data.frame(train$x),y=as.vector(train$y),kernel="linear",type="C-classification",cost=cvec[j])
       pred <- predict(model, data.frame(validation$x))
      validation.err[i,j]= sum(pred!=validation$y)/length(pred)
    }}
  
  
  for (j in 1:nc) {
    print(paste("C=",cvec[j]," : validation.error=",mean(validation.err[,j]),"+-",sd(validation.err[,j])))
  }
  merr= colMeans(validation.err)
  opt.j=which(merr<=min(merr)+1e-10)[1];
  return(cvec[opt.j])
}

prob1.2 <- function(trn,tst, cvec) {
  print("=== test error ===")
  nc=length(cvec)
  for (j in 1:nc) {
    model<-svm(x=data.frame(trn$x),y=as.vector(trn$y),kernel="linear",type="C-classification",cost=cvec[j])
    pred <- predict(model, data.frame(tst$x))
    test.err= sum(pred!=tst$y)/length(tst$y)
    print(paste("C=",cvec[j]," : test.error=",test.err))
  }
}

prob1.3 <- function(trn,tst,Copt,fpr.end) {
  print("=== roc ===")

  model<-svm(trn$x,as.vector(trn$y),kernel="linear",type="C-classification",cost=Copt,probability = TRUE)
  pred<-predict(model, tst$x,probability=TRUE)
  pred.score<-attr(pred,"probabilities")[,1]
  
  roc.svm=roc.points(tst$y,pred.score)

 
 model=glm.fit(trn$x,trn$y, family = binomial())
 pred.score=tst$x%*%as.matrix(model$coef)
 roc.logistic=roc.points(tst$y,pred.score)
 

 # call least squares and compute score on test set
 model=glm.fit(trn$x,trn$y, family = gaussian())
 pred.score=tst$x%*%as.matrix(model$coef)
 roc.ls=roc.points(tst$y,pred.score)
  
  outfn="prob1.3-roc.pdf";
  cat(paste("output plot to ",outfn,"\n"))
  pdf(file=outfn)
  fpr=roc.svm$fpr
  tpr=roc.svm$tpr
  ii=which(fpr<=fpr.end)
  plot(fpr[ii],tpr[ii],xlab="fpr",ylab="tpr", type="p",pch=4,col=1)

 fpr1= roc.logistic$fpr
 tpr1= roc.logistic$tpr
 ii=which(fpr1<=fpr.end)
 points(fpr1[ii],tpr1[ii],pch=20,col=2)
 
 fpr2=roc.ls$fpr
 tpr2=roc.ls$tpr
 ii=which(fpr2<=fpr.end)
 points(fpr2[ii],tpr2[ii],pch=5,col=3)

  legend(x="bottomright",legend=c("svm","logistic","least squares"), col=c(1,2,3),pch=c(4,20,5))
  dev.off()
}



prob.calib <- function(pred.score,y,nbuckets,outfile) {
  
  
  ind=sort(pred.score,decreasing=FALSE,index.return=TRUE)$ix
  pred.score=sort(pred.score,decreasing=FALSE)
  sorted.y=y[ind]
  n=length(pred.score)
  #split data into buckets,calculate bucket size
  bs <- ceiling(n/nbuckets)
  pr.prob<-rep(0,nbuckets)
  bp<-rep(0,n)
  for (i in 1:nbuckets) {
    
    start=1+(i-1)*bs
    end=ifelse(n>=i*bs,i*bs,n)
    bucket.start=pred.score[start]
    bucket.end=pred.score[end]
    positive=sum(sorted.y[start:end]>0)
    total=length(pred.score[start:end])
    
    bucket.prob=positive/total
    bucket.num=length(pred.score[start:end])
    # this estimate the std of the probability
    pr.prob[i]=bucket.prob
    bp[start:end]=bucket.prob
    bucket.std=sqrt(pr.prob[i]*(1.0-pr.prob[i])/bucket.num)
    print(paste("[",bucket.start,",",bucket.end,"] :",bucket.prob,"+-",bucket.std, " (",bucket.num,")"))
  }
  cat(paste("output calibrated probability to ",outfile,"\n"))
  pdf(file=outfile)
  
  
  plot(pred.score,bp,xlab="predict score",ylab="P(Y=1|X)", type="p",pch=1,col=3)
  dev.off()
 
}

prob1.4 <- function(trn,tst,Copt,nbuckets) {
  print("=== probability calibration with svm ===")
  
  # call svm and compute predicted scores on test set
  model<-svm(trn$x,as.vector(trn$y),kernel="linear",type="C-classification",cost=Copt,probability = TRUE)
  pred<-predict(model, tst$x,probability=TRUE)
  pred.score<-attr(pred,"probabilities")[,1]
  prob.calib(pred.score,tst$y,nbuckets,"prob1.4-svm.pdf")

  print("=== probability calibration with logisitc regression ===")
  
  # call logistic regression and compute predicted probability on test set
  model=glm.fit(trn$x,trn$y, family = binomial())
  pred=exp(-(tst$x%*%as.matrix(model$coef)))
  pred.prob=1/(1+pred)
  prob.calib(pred.prob,tst$y,nbuckets,"prob1.4-logistic.pdf")

  print("=== probability calibration with least squares regression ===")

  # call least squares and compute predicted probability on test set
  model=glm.fit(trn$x,trn$y, family = gaussian())
  pred=tst$x%*%as.matrix(model$coef)
  score=(pred+1)/2
  pred.prob=rep(0,length(score))
  for(i in 1:length(score)){
  pred.prob[i]=max(0,min(1,score[i]))
  }
  prob.calib(pred.prob,tst$y,nbuckets,"prob1.4-ls.pdf")

}


trn=read.data('wdbc.train')
tst=read.data('wdbc.test')

cvec=c(100,32,10,3.2,1,0.32,0.1,0.032,0.01,0.0032,0.001,0.00032,0.0001)

Copt=prob1.1(trn,cvec,10,0.8)

prob1.2(trn,tst,cvec)

prob1.3(trn,tst,Copt,0.25)

prob1.4(trn,tst,Copt,10)






