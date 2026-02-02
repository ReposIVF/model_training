import os
import shutil


def clear_folder(objectId):
    # Ruta del folder a eliminar
    print(f"cleaning files in folder: [{objectId}]")

    folder_path = f"./{objectId}"

    # Eliminar el directorio
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Directorio '{folder_path}' eliminado exitosamente.")
        else:
            print(f"El directorio '{folder_path}' no existe, creando uno nuevo.")
    except FileNotFoundError:
        print(f"El directorio '{folder_path}' no existe.")
    except OSError:
        print(f"El directorio '{folder_path}' no está vacío o no puede ser eliminado.")



def clean_files(downloaded_files):
    """
        description: clean files from the /tmp directory.
        
        params:
            - downloaded_files (list): list of files to delete.
        
        returns:
            - None
    """
    try:
        for file in downloaded_files:
            print(f"Cleaning file: {file}")
            os.remove(file)
        
    except Exception as e:
        raise Exception(f"Error cleaning files: {e}")
    

def clean_files2(embryos_list):
    """
        description: clean files from the /tmp directory.
        
        params:
            - downloaded_files (list): list of files to delete.
        
        returns:
            - None
    """
    print("\n \n \n")

    try:
        for file in embryos_list:
            print(f"Cleaning embryo: {file['embryo']}")
            os.remove(file['image'])
            os.remove(file['image_cropped'])
            os.remove(f"./{file['image_processed']}")
        
    except Exception as e:
        raise Exception(f"Error cleaning files: {e}")