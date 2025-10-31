from abc import abstractmethod
import os
from pathlib import Path


class CourseFolder:
    def __init__(
            self,
            course_code,
            ):
        self.__course_code = str(course_code)
        self.__data_folder_name = "data"
        self.__root_path = Path.cwd()
        self.__rqmts_url = "https://raw.githubusercontent.com/aso-uts/labs_datasets/main/36106-mlaa/requirements.txt"
        self.folder_dir = None
        self.folder_path = None

    @property
    def course_code(self):
        return self.__course_code
    
    @property
    def data_folder_name(self):
        return self.__data_folder_name
    
    @property
    def root_path(self):
        return self.__root_path
    
    @property
    def rqmts_url(self):
        return self.__rqmts_url
    
    def exec_pip_install(self):
        #print("###### Install required Python packages ######")
        os.system(f'pip install -q -r {self.__rqmts_url}')

    def exec_ignore_warnings(self):
        #print("###### Install required Python packages ######")
        os.system('import warnings')
        os.system('warnings.filterwarnings("ignore")')
        #os.system('warnings.simplefilter(action="ignore", category=FutureWarning)')
    
    @property
    def is_colab(self):
        return os.getenv("COLAB_RELEASE_TAG")

    def mount_gdrive(self):
        if self.is_colab:
            from google.colab import drive

            #print("\n###### Connect to personal Google Drive ######")
            gdrive_path = "/content/gdrive"
            drive.mount(gdrive_path)
            self.__root_path = Path(f"{gdrive_path}/MyDrive/")

    def create_folders(self):
        #print("\n###### Setting up folders ######")
        if self.folder_dir:
            self.folder_path = self.__root_path / self.folder_dir
            self.folder_path.mkdir(parents=True, exist_ok=True)

            print(f"\nYou can now save your data files in: {self.folder_path}")

    def run(self):
        self.exec_pip_install()
        self.exec_ignore_warnings()
        self.mount_gdrive()
        self.create_folders()
        if self.is_colab:
            os.system(f'cd {self.folder_path}')


class LabExFolder(CourseFolder):
    def __init__(
            self,
            course_code,
            lab,
            exercise,
            ):
        CourseFolder.__init__(self, course_code=course_code)
        self._lab = lab
        self._exercise = exercise
        self.__labs_folder_name = "labs"
        self._lab_dir = f"{self.course_code}/{self.__labs_folder_name}/{self._lab}/{self._exercise}"
        self.folder_dir = f"{self._lab_dir}/{self.data_folder_name}"


class AtFolder(CourseFolder):
    def __init__(
            self,
            course_code,
            assignment,
            ):
        CourseFolder.__init__(self, course_code=course_code)
        self._assignment = assignment
        self.__at_folder_name = "assignment"
        self._at_dir = f"{self.course_code}/{self.__at_folder_name}/{self._assignment}"
        self.folder_dir = f"{self._at_dir}/{self.data_folder_name}"