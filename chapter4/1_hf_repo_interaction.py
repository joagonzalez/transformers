"""
First login to hugging face using a token generated @ https://huggingface.co/settings/tokens:

# huggingface-cli login 
"""

import sys
from huggingface_hub import upload_file, delete_file, Repository


if len(sys.argv) == 1:
    print('invalid input!')
    exit()
    
if sys.argv[1] == 'upload':
    print('uploading file')
    upload_file(
        path_or_fileobj='test.py', 
        path_in_repo='test.py', 
        repo_id='joagonzalez/mim-tesis-whisper',
        commit_message='initial commit'
        )   
elif sys.argv[1] == 'delete':
    print('deleting file')
    delete_file(
        path_in_repo='test.py', 
        repo_id='joagonzalez/mim-tesis-whisper',
        commit_message='delete file'
        )
elif sys.argv[1] == 'repo':
    print('instantiating hf repository')
    repo = Repository(
        local_dir='hf-whisper',
        clone_from='joagonzalez/mim-tesis-whisper'
    )
    
    # test push model metadata and weights
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
   
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    repo.git_pull()
    
    model.save_pretrained(repo.local_dir)
    tokenizer.save_pretrained(repo.local_dir)
    
    print('push changes to repo')
    
    repo.git_add()
    repo.git_commit('added model and tokenizer')
    repo.git_push()
    

else:
    print('invalid input!')