from search_child_node import get_child_chunks
from re_ranking_node import cross_encoder_re_rank, reciprocal_rank_fusion
from search_parent_node import get_parent_chunks
from contextual_compressor_node import compress_context


# query = "What did the people ask Allah's Messenger about seeing their Lord on the Day of Resurrection?"
query = "What are the six articles of Faith mentioned in the document?"
# query = "Can you provide details about the last action of the Prophet mentioned in Hadith H.732?"
# query = "What does the Prophet say about people who have eaten garlic or onion before coming near the mosque?"

inputs = {"keys": {"query": query, "generated_queries": []}}
child_dox = get_child_chunks(inputs)
print(f'Child: {len(child_dox["keys"]["child_dox"])}\n')
# parent_ids = reciprocal_rank_fusion(child_dox)
# print(f'Reciprocal Parent Ids: {parent_ids["keys"]["parent_ids"]}\n')
parent_ids = cross_encoder_re_rank(child_dox)
print(f'Cross Parent Ids: {parent_ids["keys"]["parent_ids"]}\n')
parent_docs = get_parent_chunks(parent_ids)
print(f'Parent Docs: {len(parent_docs["keys"]["parent_docs"])}\n')
compressed_docs = compress_context(parent_docs)
print(f'Compressed Docs: {len(compressed_docs["keys"]["compressed_docs"])}\n')
