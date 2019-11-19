library(haven)
library(tidyverse)
library(readxl)
library(stringr)
library(ggplot2)
library(GGally)
library(modelr)
library(gridExtra)
library(grid)
church<-read_dta('dataset_ARDA.dta')
county<-read_excel('dataset_uscb.xlsx')
church1 <- church%>%
  group_by(stateab,fipsmerg,cntynm,year,family)%>%
  filter(!is.na(adherent))%>%
  filter(!str_detect(grpname, 'Combine', negate = FALSE)&!str_detect(grpname, 'old count', negate = FALSE))%>%
  filter(totpop!=0&cntynm!='Loving County')%>%
  summarise(n=sum(adherent,na.rm=T)) %>%
  filter(n == max(n)) %>% 
  ungroup()%>% 
  dplyr::select(-fipsmerg)
test<-church%>%
  group_by(stateab,cntynm,year)%>%
  filter(!is.na(adherent))%>%
  filter(!str_detect(grpname, 'Combine', negate = FALSE)&!str_detect(grpname, 'old count', negate = FALSE))%>%
  filter(totpop!=0&cntynm!='Loving County')%>%
  summarize(countad=sum(adherent),totpop=totpop[1],STCOU=fipsmerg[1])%>%
  mutate(rate1=countad/totpop*100,STCOU=as.character(STCOU))%>%
  mutate(STCOU=ifelse(nchar(STCOU)==4,paste0('0',STCOU),STCOU))%>%
  left_join(county,by=c('STCOU'='STCOU'))%>%
  filter(year==Year)%>%
  select(-c(Year,`Male population ALL YEARS (complete count)`, `Female population 1980 (complete count)`))%>%
  rename(highschool='High-school Only',bachelor='Educational attainment - persons 25 years and over - percent bachelor\'s degree or higher')%>%
  rename(income='Median household income in 1979',highincome='Households with income of $75,000 or more in 1979')%>%
  left_join(church1,by=c('stateab','cntynm','year'))
test1<-test%>%
  rename(young='Resident population under 18 years (April 1 - complete count)',
         old1='Resident population 55 to 59 years (complete count) 1980',
         old2='Resident population 60 to 64 years (complete count)' ,
         old3='Resident population 65 years and over (complete count)')%>%
  mutate(workingrate=(totpop-old1-old2-old3-young)/totpop,rate2=n/totpop*100,income=income/100)%>%
  mutate(rate2=ifelse(rate2>100,100,rate2))%>%
  select(stateab,cntynm,totpop,year,income,highincome,highschool,bachelor,workingrate,rate2)%>%
  filter(income!=0)
test1980_<-test1%>%
  filter(year==1980)
test1990_<-test1%>%
  filter(year==1990)
test2000_<-test1%>%
  filter(year==2000)
test2010_<-test1%>%
  filter(year==2010)
my_model<-function(df){
  model1<-lm(log10(income)~rate2,data=df)
  model2<-lm(highschool~rate2,data=df)
  model3<-lm(income~rate2,data=df)
  model4<-lm(log10(income)~rate2,data=df,weights=1/(rate2*(100+0.01-rate2)/totpop))
  list(model1,model2,model3,model4)
}
theme_set(theme_light())
mutatedata<-function(df,mod){
  plotdf<-df%>%
    add_residuals(mod)%>%
    add_predictions(mod)%>%
    ungroup%>%
    mutate(SDR=rstudent(mod))
  return(plotdf)
}
drawgrid<-function(plotdf,x1,y1){
  g1<-ggplot(plotdf,aes(x=plotdf[[x1]],y=plotdf[[y1]]))+
    geom_point(alpha=0.3)+
    geom_line(aes(y=pred),col='red')+
    labs(x=x1,
         y=y1)
  
  g7<-ggplot(plotdf,aes(x=plotdf[[x1]],y=plotdf[[y1]]))+
    geom_point(alpha=0.3)+
    geom_line(aes(y=10^pred),col='red')+
    labs(x=x1,
         y=y1)
  g2<-ggplot(plotdf,aes(sample=SDR))+
    geom_qq()+
    geom_qq_line()+
    labs(title='qq plot')
  g3<-ggplot(plotdf,aes(x=SDR))+
    geom_histogram(binwidth=0.2)+
    labs(title='histogram')
  g4<-ggplot(plotdf,aes(y=SDR,x=1:nrow(plotdf)))+
    geom_line()+
    geom_point(shape=1)+
    labs(title='line_plot')
  g5<-ggplot(plotdf,aes(y=SDR,x=pred))+
    geom_point(shape=1)+
    geom_smooth(color='red',se=FALSE)+
    labs(title='SDR verses y_hat',
         x='y_hat')
  g6<-ggplot(plotdf,aes(y=SDR,x=plotdf[[x1]]))+
    geom_point(shape=1)+
    geom_smooth(color='red',se=FALSE)+
    labs(title='SDR verses x',
         x=x1)
  list(g1,g2,g3,g4,g5,g6,g7)
}
#2010
mod2010<-my_model(test2010_)[[4]]
testplot2010<-mutatedata(test2010_,mod2010)
g<-drawgrid(testplot2010,'rate2','income')
grid.arrange(g[[1]],g[[2]],g[[3]],g[[4]],ncol=2)
modellog<-lm(log10(income)~rate2,data=test2010_)
testplot1980<-mutatedata(test2010_,modellog)
grid<-tibble(x=seq(0,120,length.out=1000),y=10^(2.651181-0.001044*seq(0,120,length.out=1000) ))
ggplot(testplot1980,aes(x=rate2,y=income))+
  geom_point(alpha=0.3)+
  geom_line(grid,mapping=aes(x=x,y=y))

ggplot(testplot1980,aes(sample=resid))+
  stat_qq()+
  stat_qq_line()
#1980

mod1980<-my_model(test1980_)[[1]]
testplot1980<-mutatedata(test1980_,mod1980)
g80<-drawgrid(testplot1980,'rate2','income')
grid.arrange(g80[[7]],g80[[2]],g80[[4]],g80[[5]],ncol=2)
#1990

mod1990<-my_model(test1990_)[[1]]
testplot1990<-mutatedata(test1990_,mod1990)
g90<-drawgrid(testplot1990,'rate2','income')
grid.arrange(g90[[7]],g90[[2]],g90[[4]],g90[[5]],ncol=2)
#2000

mod2000<-my_model(test2000_)[[1]]
testplot2000<-mutatedata(test2000_,mod2000)
g00<-drawgrid(testplot2000,'rate2','income')
grid.arrange(g00[[7]],g00[[2]],g00[[4]],g00[[5]],ncol=2)
#1980

testplot1980e<-mutatedata(test1980_,my_model(test1980_)[[2]])
gg80<-drawgrid(testplot1980e,'rate2','highschool')
grid.arrange(gg80[[1]],gg80[[2]],gg80[[4]],gg80[[5]],ncol=2)
#1990

testplot1990e<-mutatedata(test1990_,my_model(test1990_)[[2]])
gg90<-drawgrid(testplot1990e,'rate2','highschool')
grid.arrange(gg90[[1]],gg90[[2]],gg90[[4]],gg90[[5]],ncol=2)
#2000


testplot2000e<-mutatedata(test2000_,my_model(test2000_)[[2]])
gg00<-drawgrid(testplot2000e,'rate2','highschool')
grid.arrange(gg00[[1]],gg00[[2]],gg00[[4]],gg00[[5]],ncol=2)
#2010


testplot2010e<-mutatedata(test2000_,my_model(test2010_)[[2]])
gg10<-drawgrid(testplot2000e,'rate2','highschool')
grid.arrange(gg10[[1]],gg10[[2]],gg10[[4]],gg10[[5]],ncol=2)
