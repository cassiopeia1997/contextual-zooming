import numpy as np
from util_functions import vectorize
import math
import random
import datetime
import os.path
from conf import sim_files_folder, save_address
class BallStruct:
	def __init__(self, featureDimension,number,r,x,y,T):
		self.d = featureDimension
		self.x = x
		self.y = y
		self.r = r
		self.number = number
		self.nt=0
		self.T=T
		self.rew = 0.0
	def getdistance(self, CoTheta,y):
		disX = CoTheta - self.x
		disY= y - self.y
		c=np.hstack((disX,disY))
		dis= np.linalg.norm(c,ord=2)
		#print('dis',dis**2)
		#print(np.linalg.norm(disX,ord=2)**2)
		#print(np.linalg.norm(disY,ord=2)**2)
		return dis
	def getpre(self):
		#rew=self.rew
		#nt=self.nt
		#self.rew+=1
		vt=self.rew/max(1,self.nt)
		pre=vt+self.getconf()+self.r

		return pre
	def getconf(self):
		conf=4*np.sqrt(float(math.log10(self.T))/(1+self.nt))
		return conf
	
	def updateCounters(self,reward):
		self.nt += 1
		self.rew += reward
		#print('reward',self.rew)
		#print('nt',self.nt)




#---------------LinUCB(fixed user order) algorithm---------------
class CZoomingAlgorithm:
	def __init__(self, dimension, alpha, lambda_,T):  # n is number of users
		self.balls = []
		self.dimension = dimension
		r=1
		x=np.zeros(dimension)
		y=np.zeros(dimension)
		self.T=T
		self.balls.append(BallStruct(dimension,0,r,x,y,self.T))
		self.CanEstimateUserPreference = False
		self.CanEstimateCoUserPreference = False
		self.CanEstimateW = False
		self.CanEstimateV = False
		self.time=0
		self.startTime = datetime.datetime.now()
		timeRun = self.startTime.strftime('_%m_%d_%H_%M') 
		self.filenameWritePara=os.path.join(save_address, '_'+'BallNum'+timeRun+'.cluster')
	def decide(self, pool_articles, userID, userIDCoTheta):
		minb = float('inf')
		index=float('-inf')
		articlePicked = None
		ballPicked = None
		relevant=[]
		Btoy={}
		for i in self.balls:
			Btoy[i.number]=[]
		for y in pool_articles:
			feature=y.contextFeatureVector[:self.dimension]
			for b in self.balls:
			
				dis1 = b.getdistance(userIDCoTheta,feature)
				#print dis1,b.r
				if dis1 <= b.r:
					relevant.append(b)
					Btoy[b.number].append(y)
					#print(relevant)
					for k in self.balls:
						if (k.r<b.r) &(k.getdistance(userIDCoTheta,feature)< k.r):
							
							relevant.remove(b)
							
							Btoy[b.number].remove(y)
							#print("true")
							break;
							
					'''
					dis2 = balls[self.parent[b.number][0]].getdistance(userIDCoTheta,feature)
					if dis2 > balls[self.parent[b.number][0]].r:
						relevant.append(b)
						if (type(Btoy[b.number])==list):
							Btoy[b.number].append(y)
						else:
							Btoy[b.number]=[]
							Btoy[b.number].append(y)
			'''	
		#print(relevant)
		for i in relevant:
			#print('i')
			for b in self.balls:
				#print('o')
				if i!=b:
					temp=b.getpre()+b.getdistance(i.x,i.y)
					if temp < minb:
						minb = temp
			indexnow=i.r+minb
			if indexnow > index:
				#print('true')
				ballPicked=i
				index=indexnow
		articlePicked=random.choice(Btoy[ballPicked.number])
		

	

		return [articlePicked,ballPicked]
	def updateCounters(self,reward,ballPicked):
		ballPicked.updateCounters(reward)
		self.time+=1
		
	def updateBall(self, ballPicked,Cotheta,articlePicked):
		b=ballPicked
		
		if (b.getconf()<=b.r):
			sumb=len(self.balls)
			self.balls.append(BallStruct(self.dimension,sumb,0.5*b.r,Cotheta,articlePicked.contextFeatureVector[:self.dimension],self.T))
		#print(len(self.balls))
		with open(self.filenameWritePara, 'a+') as f:
			f.write(str(self.time/30)+'\t'+str(len(self.balls)))
			f.write('\t'+str(ballPicked.number)+'\t'+str(articlePicked.id)+'\t')
			f.write('\n')		
		#print(b.nt,b.getconf(),b.r)
		

