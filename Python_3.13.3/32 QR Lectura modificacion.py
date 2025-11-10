import cv2
import qrcode
from PIL import Image
import matplotlib.pyplot as plt

# Paso 1: Leer imagen QR original
ruta_entrada = r"C:\Users\Alvaro\OneDrive\Documentos\Python\Curso Python\datasets\WireGuard.jpg"
img = cv2.imread(ruta_entrada)
detector = cv2.QRCodeDetector()
data, _, _ = detector.detectAndDecode(img)

# Paso 2: Mostrar contenido original
print("🔍 Contenido original:")
print(data)

# Paso 3: Reemplazar IP por dominio
nuevo_data = data.replace(
    "Endpoint = 152.202.14.230:51820",
    "Endpoint = nicolas1.tplinkdns.com:51820"
)

print("\n✅ Contenido modificado:")
print(nuevo_data)

# Paso 4: Generar nuevo QR
qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_Q,
    box_size=10,
    border=4
)
qr.add_data(nuevo_data)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")

# Paso 5: Mostrar el nuevo QR
plt.figure(figsize=(6, 6))
plt.imshow(img_qr, cmap="gray")
plt.axis("off")
plt.title("Nuevo QR con dominio actualizado")
plt.show()

# Guardar nuevo QR en carpeta datasets
ruta_salida = r"C:\Users\Alvaro\OneDrive\Documentos\Python\Curso Python\datasets\WireGuard_QR_actualizado.png"
img_qr.save(ruta_salida)

# Confirmación en consola
print(f"\n✅ Código QR guardado exitosamente en:\n{ruta_salida}")

# Paso 6: Leer el QR generado y mostrar su contenido
img_generado = cv2.imread(ruta_salida)
data_generado, _, _ = detector.detectAndDecode(img_generado)
print("\n🔄 Contenido leído del QR generado:")
print(data_generado)