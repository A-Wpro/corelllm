
import gc
import torch
from torch import inf
from torch import cuda, bfloat16
import transformers

gc.collect()
torch.cuda.empty_cache()

model_id = '/shared/data1/Users/l1058760/Llama-2-70b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)

# enable evaluation mode to allow model inference
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id
)
generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=2048,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)
"""
# checking again that everything is working fine
p = llm(prompt="Explain me the Oil & Gas industry.")
print(p)
"""

from langchain.document_loaders import JSONLoader
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["dttmstart"] = record.get("dttmstart")
    metadata["dttmend"] = record.get("dttmend")
    metadata["jbr_dict"] = record.get("jbr_dict")
    metadata["jobsubtyp"] = record.get("jobsubtyp")

    return metadata
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


user_input = input("Please enter a well or ALL: ")

try:	
	if user_input != "ALL":
		user_input = user_input+"_JLI"
	n = f"./FAISS/faiss_{user_input}"
	vectorstore = FAISS.load_local(n, embeddings)
except:
	FAISS_files = glob.glob('./FAISS/faiss_*')
	if len(FAISS_files) == 0 or "/FAISS/faiss_ALL" not in FAISS_files  :
		json_files = glob.glob('./data/*.json')
		for name in json_files:
			print(name)
			loader = JSONLoader(
			    file_path=f'./{name}',
			    jq_schema='.[].jbr_dict',
			    text_content=False,
			    metadata_func=metadata_func)

			documents = loader.load()
			

			text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
			all_splits = text_splitter.split_documents(documents)



			# storing embeddings in the vector store
			db_tmp = FAISS.from_documents(all_splits, embeddings)
			n = "./FAISS/faiss_"+str(name.split("/")[-1].split(".")[0])
			db_tmp.save_local(n)
			if "db" not in locals():
				vectorstore = db_tmp
			else:
				vectorstore.merge_from(db_tmp)
		vectorstore.save_local("./FAISS/faiss_ALL")
	try:	
		n = f"./FAISS/faiss_{user_input}"
		if user_input != "ALL":
			n = n+"_JLI"
		vectorstore = FAISS.load_local(n, embeddings)
	except:
		vectorstore = FAISS.load_local("./FAISS/faiss_ALL", embeddings)

from langchain.chains import ConversationalRetrievalChain
##--
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
chat_history = []

query = "What can you say about DFF22?"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])

print()
chat_history = [(query, result["answer"])]
FAISS_files = glob.glob('./FAISS/faiss_*')
FAISS_files = list(map(lambda x: x.split("_")[1], FAISS_files))

while user_input != "end":
	user_input = input("Please enter something: ")
	for w in user_input: 
		if w in FAISS_files:
			n = f"./FAISS/faiss_{w}"
			vectorstore.merge_from(FAISS.load_local(n, embeddings))
	if user_input != "end":
		query = user_input
		result = chain({"question": query, "chat_history": chat_history})
		chat_history = [(query, result["answer"])]
		print(result['answer'])
		print(result['source_documents'])
		print("-"*100)
		print()
