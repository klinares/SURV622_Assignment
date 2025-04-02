

# Load the rollama package and ping Ollama to ensure connectivity.
library(rollama)
library(tictoc)
library(readxl)
library(writexl)
library(tidyverse)

ping_ollama()


# download models
pull_model("llama3.2:3b")
pull_model("deepseek-r1:8b")



# read in data
comments <- read_xlsx(
  "D:/repos/SURV622_Assignment/data/comments_to_code/merged_codes.xlsx") 


# provide prompt
instructions <- "Instruction: You have assumed the role of a stakeholder that is presented 
  with a reddit comments from likely federal workers
  related to the current polcieis on reducing the federal workforce. 
  Please determine the author of the comment's stance on this topic, and only provide the answer."

prompt <- "Is this comment in 'favor', 'neutral', or 'oppose' the reduction in federal workforce? 
Provide one word answer only!"



# zero shot query generation
options(rollama_model = "deepseek-r1:8b")

# run the model through the comments and time it
tic() # start timer for task

dat <- comments |> 
  # create a new variable to save answers from llama
  mutate(
    
    llama =
        # provide query to the local instance of llama
           query(make_query(
             text = comment, # pass each reddit comment
             prompt = prompt, # provide same prompt
             system = instructions), # provide task
             model = "llama3.2:3b", # pass model
             output = "text"), # pull answer out
        
    deepseek =
       # provide query to the local instance of llama
       query(make_query(
         text = comment, # pass each reddit comment
         prompt = prompt, # provide same prompt
         system = instructions), # provide task
         model = "deepseek-r1:8b", # pass model
         output = "text") # pull answer out
  )
toc() # end of timer


# clean up responses to only use the categories we need
#`%not_in%` <- purrr::negate(`%in%`)

new_dat <- dat |> 
  mutate(
    # convert to lower cases
    llama = str_to_lower(llama),
    deepseek = str_to_lower(deepseek),
    # take out periods
    llama = str_remove_all(llama, "\\."),
    # extract anwser from deepseek
    deepseek = str_sub(deepseek, start= -12),
    deepseek = str_extract(deepseek, "\\s*([^\\s]+)$"),
    deepseek = str_trim(deepseek),
    # keep answers that are needed
    llama = case_when(
      str_detect(llama, "favor") ~ "favor",
      str_detect(llama, "oppose") ~ "oppose",
      str_detect(llama, "neutral") ~ "neutral",
      TRUE ~ NA_character_),
    deepseek = case_when(
      str_detect(deepseek, "favor") ~ "favor",
      str_detect(deepseek, "oppose") ~ "oppose",
      str_detect(deepseek, "neutral") ~ "neutral",
      TRUE ~ NA_character_) 
  )

# write to gibhub
write_xlsx(new_dat,
  "D:/repos/SURV622_Assignment/outputs/reddit_comments_LLM_analysis.xlsx") 
