import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("/home/kewei/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/b135031bae30b5ac2ae141a0e68717795ce38340/model.pt").to(device)                             

ret, frame = cap.read()

while ret:
    # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

    # Infer 
    output = model.infer(input_image)

    depth_map = output['depth'].cpu().numpy()
    cv2.imshow('Depth Map', depth_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()