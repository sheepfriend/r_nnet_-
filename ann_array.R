g<-function(x){return(1/(1+exp(-x)))}

alpha<-0.001
scale<-1

nnet_init<-function(attr.in,num.nodes,attr.out,cross.num){
	result<-list()
	result$attr.in<-attr.in
	result$attr.out<-attr.out
	result$num.nodes<-num.nodes
	result$cross.num<-cross.num
	result$layers<-list()
	for(i in 1:length(num.nodes)){
		if(i==1){
			result$layers[[i]]<-list()
			result$layers[[i]]$weight<-array((runif(num.nodes[i]*(attr.in+1)*cross.num)-0.5)*scale,dim=c(num.nodes[i],attr.in+1,cross.num))
		}
		else{
			result$layers[[i]]<-list()
			reuslt$weight<-array((runif(num.nodes[i]*(attr.in+1)*cross.num)-0.5)*scale,dim=c(num.nodes[i],num.nodes[i-1]+1,cross.num))
		}
	}
	result$out<-list()
	result$out$weight<-array((runif(num.nodes[i]*(attr.in+1)*cross.num)-0.5)*scale,dim=c(attr.out,tail(num.nodes,1)+1,cross.num))
	return(result)
}


nnet_load_input<-function(result,input){
	num<-result$cross.num
	len<-floor(length(input[,1])/num)*num
	len_one<-len/num*(num-1)
	
	result$input<-input[1:len,]
	result$layers[[1]]$input<-array(0,dim=c(len_one,1+result$attr.in,num))
	result$layers[[1]]$input[,1,]<-1
	result$layers[[1]]$input[,-1,]<-rep(result$input,num-1)
	for(i in 1:length(result$num.nodes)){
		result$layers[[i]]$output<-array(0,dim=c(len_one,result$num.nodes[i],num))
		if(i>1){
			result$layers[[i]]$input<-array(0,dim=c(len_one,result$num.nodes[i]+1,num))
			result$layers[[i]]$input[,1,]<-1
			result$layers[[i]]$input[,-1,]<-result$layers[[i-1]]$output
		}
		for(j in 1:num){
			result$layers[[i]]$output[,,j]<-g(result$layers[[i]]$input[,,j]%*%t(result$layers[[i]]$weight[,,j]))
		}
	}
	result$out$input<-array(0,dim=c(len_one,tail(result$num.nodes,1)+1,num))
	result$out$input[,1,]<-1
	result$out$input[,-1,]<-result$layers[[length(result$num.nodes)]]$output
	result$out$output<-array(0,dim=c(len_one,result$attr.out,num))
	for(j in 1:num){
		result$out$output[,,j]<-g(result$out$input[,,j]%*%t(result$out$weight[,,j]))
	}
	return(result)
}


nnet_train_one_iter<-function(result,output){
	
	#output layer
	num<-result$cross.num
	len<-floor(length(output[,1])/num)*num
	len_one<-len/num*(num-1)	
	
	output<-array(rep(output[1:len,],num-1),dim=c(len_one,result$attr.out,num))
	out<-result$out$output-output
	delta<-out*(1-out)*(output-out)
	Delta<-array(0,dim=c(result$attr.out,tail(result$num.nodes,1)+1,num))
	k<-array(0,dim=c(len_one,tail(result$num.nodes,1),num))
	for(j in 1:num){
		Delta[,,j]<-(t(delta[,,j])%*%result$out$input[,,j])
		k[,,j]<-delta[,,j]%*%result$out$weight[,-1,j]
	}
	result$out$weight<-result$out$weight+Delta*2*alpha
	
	
	#other layers
	for(i in length(result$num.nodes):1){
		out<-result$layers[[i]]$output
		delta<-out*(1-out)*k
		if(i==1){
			Delta<-array(0,dim=c(result$num.nodes[i],result$attr.in+1,num))
		k<-array(0,dim=c(len_one,result$attr.in,num))
		}
		else{
			Delta<-array(0,dim=c(result$num.nodes[i],result$num.nodes[i-1]+1,num))
			k<-array(0,dim=c(len_one,result$num.nodes[i-1],num))
		}
		for(j in 1:num){
			Delta[,,j]<-(t(delta[,,j])%*%result$layers[[i]]$input[,,j])
			k[,,j]<-delta[,,j]%*%result$layers[[i]]$weight[,-1,j]
		}
		result$layers[[i]]$weight<-result$layers[[i]]$weight+Delta*2*alpha
	}
	
	return(result)
}

nnet_train<-function(result,output,iter){
	
	num<-result$cross.num
	len<-floor(length(output[,1])/num)*num
	len_one<-len/num*(num-1)	
	
	for(i in 1:iter){
		result<-nnet_load_input(result,result$input)
		result<-nnet_train_one_iter(result,output)
	}
	output<-array(rep(output,num-1),dim=c(len_one,result$attr.out,num))

	rss<-apply((output-result$out$output)^2,3,sum)
	min.loss<-min(rss)
	min.net<-which(rss==min.loss)
	result$min.loss<-min.loss
	result$min.net<-min.net
	print(min.loss)
	return(result)
}

nnet_predict<-function(result,input,cla=F){
	a<-nnet_load_input(result,input)
	out<-a$out$output[,,result$min.net]
	if(cla){
		if(length(dim(out))==1){return(0+(out>0.5))}
		output<-rep(0,length(out[,1]))
		ratio<-1/(length(out[1,])+1)
		for(i in 1:length(out[1,])){
			output[which(out[,i]>ratio)]<-i
		}
		return(output)
	}
	else{
		return(out)
	}
}

nnet_import<-function(data,y,layers=c(5),iter=100,net.max=10){
	m<-levels(as.factor(y))
	output<-c()
	for(i in 1:length(m)){
		output<-cbind(output,(y==m[i])+0)
	}
	output<-output[,-1]
	cat('data summary: response:',length(m),',sample size:',length(data[,1]),'\n')
	a<-nnet_load_input(nnet_init(length(data[1,]),layers,length(m)-1,net.max),data)
	a<-nnet_train(a,output,iter)
	return(a)
}


alpha<-0.003
scale<-10

m<-read.csv('data.csv',header=F)
m<-as.matrix(m)
m<-m[sample(length(m[,1]),length(m[,1])),]
for(i in 1:length(m[1,-1])){
	m[,i]<-(m[,i]-min(m[,i]))/(max(m[,i])-min(m[,i]))*100
}

system.time({result<-nnet_import(m[,-1],m[,1],layers=c(10),net.max=10,iter=3000);a<-nnet_predict(result,m[,-1],cla=T);print(a);table(a,m[,1])})
