
# main.py

from inspecao_dimensional import inspecao_dimensional
from modelo_cnn import inspecao_visual_cnn
import cv2

def inspecao_final(caminho_imagem):
    """
    Orquestra a inspeção completa (Dimensional + Visual) e dá o veredito final.
    """
    print(f"--- Iniciando Inspeção para: {caminho_imagem} ---")

    dim_aprovada, dim_msg = inspecao_dimensional(caminho_imagem)
    print(f"Status Dimensional: {'APROVADO' if dim_aprovada else 'REPROVADO'} - {dim_msg}")

    visual_aprovada, visual_msg = inspecao_visual_cnn(caminho_imagem)
    print(f"Status Visual CNN: {'APROVADO' if visual_aprovada else 'REPROVADO'} - {visual_msg}")

    if dim_aprovada and visual_aprovada:
        resultado_final = "APROVADO"
        cor = (0, 255, 0) 
    else:
        resultado_final = "REPROVADO"
        cor = (0, 0, 255)
        
    print(f"\nRESULTADO FINAL: {resultado_final}")
    
    # Exibir resultado na imagem 
    # img = cv2.imread(caminho_imagem)
    # if img is not None:
    #     cv2.putText(img, f"FINAL: {resultado_final}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        # cv2.imshow("Resultado Final", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    
    # TESTE 1: APROVADO GERAL
    CAMINHO_APROVADO = "dataset/validation/APROVADO/Arty_Top_jpg.rf.9841f504a1887511e9a28facf362d2bf.jpg"
    
    # TESTE 2: REPROVADO VISUAL
    CAMINHO_REPROVADO_VISUAL = "dataset/validation/REPROVADO/l_light_01_missing_hole_11_5_600.jpg" 
    
    # TESTE 3: REPROVADO DIMENSIONAL
    CAMINHO_REPROVADO_DIMENSIONAL = "dataset/validation/APROVADO/Zybo_jpg.rf.4d8fa7e81f8ff8a8dd00742b751b0e87.jpg"
    
    # -------------------------------------------------------------
    # EXECUÇÃO DOS TESTES
    # -------------------------------------------------------------
    
    print("\n--------------------------------------------------")
    print("TESTE 1: COMPONENTE PERFEITO (Esperado: APROVADO)")
    print("--------------------------------------------------")
    inspecao_final(CAMINHO_APROVADO) 
    
    print("\n--------------------------------------------------")
    print("TESTE 2: FALHA VISUAL (Esperado: REPROVADO pela CNN)")
    print("--------------------------------------------------")
    inspecao_final(CAMINHO_REPROVADO_VISUAL) 
    
    print("\n--------------------------------------------------")
    print("TESTE 3: FALHA DIMENSIONAL (Esperado: REPROVADO pelo OpenCV)")
    print("--------------------------------------------------")

    inspecao_final(CAMINHO_REPROVADO_DIMENSIONAL)