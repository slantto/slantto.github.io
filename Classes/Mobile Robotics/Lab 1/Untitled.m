Cereal19=serial('COM19')
set(Cereal19,'BaudRate',115200)
fopen(Cereal19)

Cereal2=serial('COM2')
set(Cereal2,'BaudRate',115200)
set(Cereal2,'InputBufferSize',10)
fopen(Cereal2)

b='Mobile Robotics';
c=uint8(b)

fprintf(Cereal19,'Mobile Robotics ')

fscanf(Cereal19, '%s')