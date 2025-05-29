import streamlit as st

from numpy import  float64, float32
from numpy import zeros, ones, eye, diag, array, arange, linspace, meshgrid

from numpy import pi
from numpy import exp, sin, cos, log, angle, real, imag, sqrt, conj
from numpy import abs as mag
from numpy.fft import fftshift, ifftshift, ifft2, fft2

from matplotlib.pyplot import plot, imshow, subplot, subplots
from cv2 import imread, IMREAD_GRAYSCALE

fft = lambda x: fftshift(fft2(x))
ifft = lambda x: ifft2(ifftshift(x))

imshow_mag = lambda x: imshow(mag(x), cmap='gray')
imshow_angle = lambda x: imshow(mag(x), cmap='gray')

from numpy.random import RandomState
prng = RandomState(42)  # for reproducibility
rand = prng.rand
randn = prng.randn
col1, col2 = st.columns([1, 3]) 
select_image = col1.selectbox("Image", ('flower', 'camera man', 'rain'))
file_names = {
    'flower':'./files/flower.jpeg',
    'camera man':'./files/cameraman.png',
    'rain':'./files/rain.png'
}


img = imread(file_names[select_image], IMREAD_GRAYSCALE)
img = img.astype(float64)/256

# create a phse object
phase_img = exp(-1j*img) 
#assuming canvas of 1 mm x 1 mm, grid of 2/512, 2/512 (mm/pixal)
N, M = img.shape

x = linspace(-1e-3, 1e-3, N) # m
y = linspace(-1e-3, 1e-3, M) # m

dx,dy = x[1] - x[0], y[1] - y[0]
# light source params
selected_laser = col1.selectbox('Select a light source: ',('HeNe (632.8 nm)',
                                                     'Ar Ion (488.0 nm)', 
                                                     'Ar Ion (514.5 nm)',
                                                     'Nd:YAG (1064 nm)',
                                                     'CO2 (10600 nm)', 
                                                     'Ruby (694.3 nm)',
                                                     'GaN (Diode Laser) (405 nm)', 
                                                     'AlGaAs (Diode Laser) (808 nm)', 
                                                     'Excimer (KrF) (248 nm)',
                                                     'Excimer (ArF) (193 nm)',
                                                     'Fiber Laser (Yb-doped, ~1030-1070 nm)'))
wavelengths = {
  "HeNe (632.8 nm)": 632.8e-9,
  "Ar Ion (488.0 nm)": 488.0e-9,
  "Ar Ion (514.5 nm)": 514.5e-9,
  "Nd:YAG (1064 nm)": 1064e-9,
  "CO2 (10600 nm)": 10600e-9,
  "Ruby (694.3 nm)": 694.3e-9,
  "GaN (Diode Laser) (405 nm)": 405e-9,
  "AlGaAs (Diode Laser) (808 nm)": 808e-9,
  "Excimer (KrF) (248 nm)": 248e-9,
  "Excimer (ArF) (193 nm)": 193e-9,
  "Fiber Laser (Yb-doped, ~1030-1070 nm)": 1050e-9
}
                         
wavelength = wavelengths[selected_laser]  # Wavelength of light (meters)
k0 = 2*pi/wavelength # wave number 1/m

# spatial frequency domain defined (limmtied by Nyquist-Shannon limit)
dkx, dky = 2/dx/N, 2/dy/M
kx = linspace(-1/dx, 1/dx - dkx, N)*(pi) #1/m
ky = linspace(-1/dy, 1/dy - dky, M)*(pi)  #1/m

# create a mish gird of shape NxM, NxM for xx,yy:
## xx = [[x[0]...x[N]], ... (M-times) [x[0]...x[N]]], 
# yy = [[y[0]... N-times y[0]] ... [y[M]... N-times y[M]]]
xx, yy = meshgrid(x,y)
# Same for kxx, kyy mesh grid
kxx, kyy = meshgrid(kx,ky)


Hz_parax = lambda kxx,kyy,z: exp(1j*z*sqrt(k0**2 - kxx**2 - kyy**2))
propgate = lambda f, h: ifft(h*fft(f))




r = (xx**2 + yy**2)
lens = lambda xx,yy, f: exp(-1j*k0*r/(2*f))

def baby_step_propgation(f, z, devides = 16):
    z_d = z/devides

    f_propgated = f
    h_z_d = Hz_parax(kxx, kyy, z=z_d)
    for i in range(devides):
        f_propgated = propgate(f_propgated, h_z_d)
    return f_propgated
circ   = xx**2 + yy**2 <(1/2*dx*N)**2




z_f = col1.slider('Propgation Distance', min_value=1e-6, max_value=100e-3, value=2e-3, step=0.5e-3)
z_b = col1.slider('Back Propgation Distance', min_value=1e-6, max_value=100e-3, value=2e-3, step=0.5e-3)

in_line_hologram = abs(baby_step_propgation(phase_img, z=z_f, devides=1))**2
in_line_hologram = in_line_hologram 
fig, ax = subplots(1,2)
ax[0].imshow(in_line_hologram, cmap='gray')
ax[0].set_title("Propgated in-line hologram")
ax[0].axis("off")
back_propgated = baby_step_propgation(in_line_hologram, z=z_b, devides=1)
ax[1].imshow(abs(back_propgated)**2 , cmap='gray')
ax[1].set_title("Recunstruction of in-line hologram")
ax[1].axis("off")

col2.markdown("""
              <style>
.big-font {
    font-size: 18px !important;
}
</style>
              """, unsafe_allow_html=True)
col2.markdown(
    f"""
    
    
    
              
   
    **Light Source** :   {selected_laser}                      
    **x, y range** :   ({x.min()*(1e3)}, {x.max()*(1e3)}) mm
    dx, dy   : {dx*1e6:.2f} $\mu $ m  
    dkx, dky   : {dkx*1e-3:.2f} /mm  
    **Sptial Frequncies Ranges**  :  ({kx.min()*1e-6:.2f}, {kx.max()*1e-6:.2f} /$\mu$ m)  
    **Propgation Distance**  : {z_f*1e3} mm                               
    
    
    
    """, unsafe_allow_html=True
    
)

col2.pyplot(fig)
