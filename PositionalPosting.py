class PositionalPosting:
    def __init__(self):
        self.frequency_in_all_documents = 0
        self.document_index_dict = {}
        self.document_frequency_dict = {}
        self.unique_documents_frequency = 0


positional_postings_lists = []
def create_positional_postings_lists(data_frame):
    print('')