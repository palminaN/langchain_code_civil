class CustomEmbedOpenRouter:

    endpoint = "https://openrouter.ai/api/v1/embeddings"
    model = "openai/text-embedding-3-small"

    def __init__(self,apikey=None):

        if apikey:
            self.apikey = apikey
            
    



    def embed_batch(self,chunks):
        response = requests.post("https://openrouter.ai/api/v1/embeddings",headers={
    "Authorization": f"Bearer {self.apikey}",
    "Content-Type": "application/json",
      },
      json={
        "model": "openai/text-embedding-3-small",
        "input": chunks
      }
                                )
    
        data = response.json()
        return data["data"]



    def embed_query(self,query:str):
        response = requests.post(
          "https://openrouter.ai/api/v1/embeddings",
          headers={
            "Authorization": f"Bearer {self.apikey}",
            "Content-Type": "application/json",
          },
          json={
            "model": "openai/text-embedding-3-small",
            "input": query
          }
        )
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        return embedding
                

    
    def embed_documents(self,chunks:str,batch_size=16):

        num_iter = len(chunks) // batch_size
        remainder = len(chunks) % batch_size

        res = []


        for i in range(num_iter):

            curr_batch = chunks[batch_size*i:batch_size*(i+1)]

            embeddings = self.embed_batch(curr_batch)
            for emb in embeddings:
                
                res.append(emb["embedding"])

        embeddings = self.embed_batch(chunks[-remainder:])
        for emb in embeddings:
                res.append(emb["embedding"])
        
        return res
