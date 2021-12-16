clear all;
close all;
addpath('../')
%Img = imread('NCC1701D.jpg');
Img = imread('Wendland.jpeg');
%Igray = rgb2gray(Img);
figure(1)
imshow(Img);
%M = double(Igray);
%save ('Wendland.txt', 'M', '-ascii');
Rchan = double(Img(:,:,1)); 
Gchan = double(Img(:,:,2));
Bchan = double(Img(:,:,3)); 
save ('Rchan.txt', 'Rchan', '-ascii');
save ('Gchan.txt', 'Gchan', '-ascii');
save ('Bchan.txt', 'Bchan', '-ascii');
%GcompressInt;
%Igray2 = mat2gray(G);
Rcompress;
Gcompress;
Bcompress;

Img2 = Img;
Img2(:,:,1) = R;
Img2(:,:,2) = G;
Img2(:,:,3) = B;

figure(2)

imshow(Img2)

%figure(3)
%surf(Img2(:,:,1)-Img(:,:,1))
%shading interp
%Rview(0,90)