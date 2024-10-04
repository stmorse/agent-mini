from utilities.faiss_utils import *

# class TestFaissManager:

#     def setUp(self):
#         self.manager = FaissManager(address=('localhost', 5001), authkey=b'faiss')
#         self.manager.connect()
#         self.index_id = 1
#         self.index_dim = 2
#         print('Connected to FaissManager')

#         self.manager.create_faiss_index(self.index_id, self.index_dim)
#         print(f'Created index {self.index_id} with dimension {self.index_dim}')

#     def test_get_index_dict(self):
#         index_dict = self.manager.get_index_dict()
#         print('Index dict: ', index_dict)
#         # assert(index_dict == {}), "Incorrect index dictionary returned"

#     def test_create_faiss_index(self):
#         self.manager.create_faiss_index(1, 2)
#         print('Index created')

#     def test_add_vectors(self):
#         vectors = [[1.0, 2.0], [3.0, 4.0]]
#         ids = [1, 2]
#         self.manager.add_vectors(self.index_id, vectors, ids)
#         print('Vectors added')
#         print(self.manager.get_index_dict())
#         # assert(len(self.manager.index) == 2), "Incorrect number of vectors added"

#     def test_search_index(self):
#         queries = [[1.0, 2.0], [3.0, 4.0], [1.5, 2.5]]
#         results = self.manager.search_index(self.index_id, queries, k=1)
#         print('Search results: ', results)
#         # assert(results[0][0] == 1), "Incorrect search result"

if __name__ == '__main__':
    # tester = TestFaissManager()
    # tester.setUp()

    # tester.test_get_index_dict()
    # tester.test_add_vectors()
    # tester.test_search_index()

    manager = get_faiss_manager(a=('faiss', 80), key=b'faiss')
    create_faiss_index(manager, 1, 2)

    vectors = [[1.0, 2.0], [3.0, 4.0]]
    add_vectors(manager, 1, vectors, [1, 2])

    queries = [[1.0, 2.0], [3.0, 4.0], [1.5, 2.5]]
    search_index(manager, 1, queries, k=1)