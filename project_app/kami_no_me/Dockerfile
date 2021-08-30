# Base image of fedora
FROM fedora:latest

# python 3.9
RUN dnf install python3 python3-wheel -y

	

# Setup the working directory
WORKDIR /app

# Copy files from the current directory to the working directory
COPY . /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN dnf -y install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm 
RUN dnf -y install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
	
RUN dnf -y install mesa-libGL 
RUN dnf -y install mesa-libGL-devel 
	
RUN dnf -y install ffmpeg 
RUN dnf -y install ffmpeg-devel
# install all dependencies from requirements.txt
#RUN --mount=type=cache,target=/root/.cache \
	#pip install -r requirements.txt

RUN pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# start server
CMD python app.py --host=0.0.0.0 --server.port=8080
