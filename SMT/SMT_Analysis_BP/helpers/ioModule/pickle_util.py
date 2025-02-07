"""
Utility functions for pickling and unpickling objects.
Generically, this is used to save and load models with doc specific to the object.
"""

import pickle as pkl
import os
import warnings
import unittest


# create a base class for pickling and unpickling
class PickleUtil(object):
    def __init__(self):
        # set calss variables to 0 to avoid errors
        self._obj = None
        self._path = None
        self._name = None
        self._docs = None

    # interior function to update the class variables
    def _update_class_vars(self, obj, path, name, docs):
        self.obj = obj
        self.path = path
        self.name = name
        self.docs = docs

    # create function to save the object
    def save(self, path: str, name: str, docs: str, obj: any):
        """
        Docstring:
        Save an object to a pickle file. The object will be saved to the path/name.pkl file. The docs will be saved to the path/name_docs.txt file.

        Parameters:
        -----------
        path: str
            The path to the directory where the object will be saved. This must be the full path to the directory.
        name: str
            The name of the pickle file. This should not include the .pkl extension.
        docs: str
            A string describing the object. This will be saved to the path/name_docs.txt file.
        obj: any
            The object to be saved. This can be any object that can be pickled.

        Returns:
        --------
        None

        Example:
        --------
        #create a PickleUtil object
        pickle_util = PickleUtil()
        #create a dict to save
        dict_to_save = {'a':1, 'b':2}
        #save the dict
        pickle_util.save(path='C:/Users/JohnDoe/Desktop', name='my_dict', docs='This is a dict I want to save', obj=dict_to_save)
        """
        # update the class variables
        self._update_class_vars(obj, path, name, docs)
        # check if the path exists, if not create it
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # create the full path to the file, add .pkl extension
        full_path = os.path.join(self.path, self.name)
        # save the object
        with open(full_path + ".pkl", "wb") as f:
            pkl.dump(self._outpkl, f)
        # save the docs to a txt file
        with open(full_path + "_docs.txt", "w") as f:
            f.write(self.docs)
        print(f"Object saved to {full_path}")

    # create a static method to load an object given a full path
    @staticmethod
    def load(full_path_to_load: str):
        """
        Docstring:
        Load an object from a pickle file. This is the full path to the pickle file.

        Parameters:
        -----------
        full_path_to_load: str
            The full path to the pickle file to load.

        Returns:
        --------
        obj: any
            The object loaded from the pickle file.

        Example:
        --------
        #create a PickleUtil object if you haven't already
        pickle_util = PickleUtil()
        #load the dict
        dict_loaded = pickle_util.load('C:/Users/JohnDoe/Desktop/my_dict.pkl')
        """
        # load the object
        with open(full_path_to_load, "rb") as f:
            pickle_loaded = pkl.load(f)
        # check if this is in the PickleUtil format (dict with "name", "docs", and "obj" keys)
        if not isinstance(pickle_loaded, dict):
            # warn the user that this is not in the PickleUtil format
            warnings.warn(
                f"Object loaded from {full_path_to_load} is not in the PickleUtil format. Returning the object as is."
            )
            return pickle_loaded
        # check if the keys are in the dict
        if all([key in pickle_loaded.keys() for key in ["name", "docs", "obj"]]):
            # print that your extracted a PickleUtil formatted object
            print(
                f'Extracted PickleUtil formatted object from {full_path_to_load} with name {pickle_loaded["name"]}'
            )
            # return the object
            return pickle_loaded

    # we need to make a dictionary explaining what the object is and what variables if any exists inside it. This will be the docs.
    # the output_obj is the object that will be saved to the pickle file
    # every time the class variables are changed, this function should be called to update the docs
    def _update_outpkl(self):
        self._outpkl = {"name": self.name, "docs": self.docs, "obj": self.obj}

    # create properties for the class variables
    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj
        self._update_outpkl()

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        "make sure this is the full path to the directory"
        if not os.path.isabs(path):
            raise ValueError("path must be absolute not relative")
        self._path = path
        self._update_outpkl()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self._update_outpkl()

    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, docs):
        self._docs = docs
        self._update_outpkl()


################## TESTS ##################


class TestPickleUtil(unittest.TestCase):
    def setUp(self):
        self.pickle_util = PickleUtil()

    def test_save_and_load(self):
        # create a dict to save
        dict_to_save = {"a": 1, "b": 2}
        # save the dict
        self.pickle_util.save(
            path=os.getcwd(),
            name="my_dict",
            docs="This is a dict I want to save",
            obj=dict_to_save,
        )
        # load the dict
        loaded_dict = self.pickle_util.load(os.path.join(os.getcwd(), "my_dict.pkl"))
        # check that the loaded dict is the same as the original dict
        self.assertEqual(dict_to_save, loaded_dict["obj"])

    # make a function to clean up the files created by the test
    def tearDown(self):
        # remove the pickle file
        os.remove(os.path.join(os.getcwd(), "my_dict.pkl"))
        # remove the docs file
        os.remove(os.path.join(os.getcwd(), "my_dict_docs.txt"))


if __name__ == "__main__":
    unittest.main()
