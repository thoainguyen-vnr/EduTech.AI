import streamlit as st
import filetype
import os

def get_file_upload(file_uploader):
    files_type = ""
    for file in file_uploader:
        root_ext  = os.path.splitext(file.name)
        st.write(root_ext[1])
    return root_ext[1]
    

def get_type_file(namefile):
    file_type = filetype.get_type(namefile)
    st.write(file_type)
    if file_type is None:
        print('Cannot guess file type!')
        return
    return file_type.extension
