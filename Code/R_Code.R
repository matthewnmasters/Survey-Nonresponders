#########################################################################################
#R script for data visualization and manipulation                                       #
#########################################################################################
setwd("C:/Users/matth/Documents/GitHub/Survey-Nonresponders/Data")
library(tidyverse)

#########################################################################################
#Reasons                                                                                #
#########################################################################################

reasons <- read.csv("modeledReasons.csv", encoding="UTF-8") %>% rename(Response = X.U.FEFF.Response)

#########################################################################################
#Manually transform to long-form                                                        #
#########################################################################################

A<- reasons %>% select(Response, TopicA, PercA) %>% rename(Topic = TopicA, Perc = PercA)
B<- reasons %>% select(Response, TopicB, PercB) %>% filter(is.na(TopicB)==FALSE) %>% rename(Topic = TopicB, Perc = PercB)
C<- reasons %>% select(Response, TopicC, PercC) %>% filter(is.na(TopicC)==FALSE) %>% rename(Topic = TopicC, Perc = PercC)

reasonsLong <- rbind(A,B,C)

reasonsRepTopics <- reasonsLong %>% group_by(Topic) %>% top_n(20, wt=Perc)

#########################################################################################
#Suggestions                                                                            #
#########################################################################################

suggestions <- read.csv("modeledSuggestions.csv", encoding="UTF-8") %>% rename(Response = X.U.FEFF.Response)

#########################################################################################
#Manually transform to long-form                                                        #
#########################################################################################

sA<- suggestions %>% select(Response, TopicA, PercA) %>% rename(Topic = TopicA, Perc = PercA)
sB<- suggestions %>% select(Response, TopicB, PercB) %>% filter(is.na(TopicB)==FALSE) %>% rename(Topic = TopicB, Perc = PercB)
sC<- suggestions %>% select(Response, TopicC, PercC) %>% filter(is.na(TopicC)==FALSE) %>% rename(Topic = TopicC, Perc = PercC)
sD<- suggestions %>% select(Response, TopicD, PercD) %>% filter(is.na(TopicD)==FALSE) %>% rename(Topic = TopicD, Perc = PercD)
sE<- suggestions %>% select(Response, TopicE, PercE) %>% filter(is.na(TopicE)==FALSE) %>% rename(Topic = TopicE, Perc = PercE)
sF<- suggestions %>% select(Response, TopicF, PercF) %>% filter(is.na(TopicF)==FALSE) %>% rename(Topic = TopicF, Perc = PercF)
sG<- suggestions %>% select(Response, TopicG, PercG) %>% filter(is.na(TopicG)==FALSE) %>% rename(Topic = TopicG, Perc = PercG)
sH<- suggestions %>% select(Response, TopicH, PercH) %>% filter(is.na(TopicH)==FALSE) %>% rename(Topic = TopicH, Perc = PercH)

suggestionsLong <- rbind(sA, sB, sC, sD, sE, sF, sG, sH)

suggestionsRepTopics <- suggestionsLong %>% group_by(Topic) %>% top_n(20, wt=Perc)

#########################################################################################
#Write the representative topics out for help naming topics                             #
#########################################################################################

write.csv(reasonsRepTopics, file="reasonsRepTopics.csv")
write.csv(suggestionsRepTopics, file="suggestionsRepTopics.csv")
