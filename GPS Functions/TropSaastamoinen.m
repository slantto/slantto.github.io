function Tropo=TropSaastamoinen(llh,Po,To,eo)
Tzd=0.002277*(1+0.0026*cos(2*llh(1))+0.00028*llh(3))*Po;
Tzw=0.0022777*((1255/To)+0.05)*eo;
Tropo=Tzd+Tzw;
end