import face_recognition
import cv2 as cv
import numpy as np
import streamlit as st


@st.cache(suppress_st_warning=True)

def main():

	with st.header("Upload the Suspected Image"):
		uploaded_file=st.file_uploader("Upload The Current jpg or png file",type=["jpg","png","jpeg","jfif"])
		
		st.image(uploaded_file,width=200)
	with st.header("Upload his Recent Image"):
		uploaded_file1=st.file_uploader("Upload his jpg or png file",type=["jpg","png","jpeg","jfif"])
		
		st.image(uploaded_file1,width=200)
	imgDhoni=face_recognition.load_image_file(uploaded_file)
	imgDhoni=cv.cvtColor(imgDhoni,cv.COLOR_BGR2RGB)
	imgTest=face_recognition.load_image_file(uploaded_file1)
	imgTest=cv.cvtColor(imgTest,cv.COLOR_BGR2RGB)

	faceLoc=face_recognition.face_locations(imgDhoni)[0]
	encodeDhoni=face_recognition.face_encodings(imgDhoni)[0]
	cv.rectangle(imgDhoni,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),3)

	faceLocTest=face_recognition.face_locations(imgTest)[0]
	encodeTest=face_recognition.face_encodings(imgTest)[0]
	cv.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


	results=face_recognition.compare_faces([encodeDhoni],encodeTest)
	facedis=face_recognition.face_distance([encodeDhoni],encodeTest)


	st.write(results)
	st.write("The Less The More Accuracy",facedis)

	cv.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
	st.image(imgTest,width=200)
	cv.imshow("Thala",imgDhoni)
	cv.imshow("test",imgTest)
	
	st.write("See the Images in the TaskBar")
	
	cv.waitKey(1)




if st.button("Thanks"):
	st.balloons()


if __name__ == '__main__':
	main()







