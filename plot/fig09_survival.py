import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def adjust_lightness(color, amount=1.0):
    """
    주어진 색상의 밝기를 조절합니다.
    amount > 1 이면 더 밝게, amount < 1 이면 더 어둡게 만듭니다.
    """
    try:
        c = plt.cm.colors.to_rgb(color)
        # RGB를 HLS로 변환
        h, l, s = colorsys.rgb_to_hls(*c)
        # 밝기(lightness) 조절
        new_l = max(0, min(1, amount * l))
        # HLS를 다시 RGB로 변환
        return colorsys.hls_to_rgb(h, new_l, s)
    except ValueError:
        return color

def create_plots():
    """
    데이터를 읽어와서 PDF로 플롯을 생성하고 저장합니다.
    """
    anions = ['beti', 'fsi', 'tfsi']
    temperatures = [298, 353, 373, 423]
    styles = {'soft': 'x11_loyal', 'hard': 'x22_loyal'}
    
    base_path = '/nas_2/transcendence/_delete/cowork/my_work/code/E_algo_final/250315_f_fit/hist'
    output_dir = '/nas_2/transcendence/revision/exports/submission_package/main/Images/f_function'
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 색상 설정
    base_colors = ["#5555FF", "#55FF55", "#FFAA55", "#FF5555"]

    for anion in anions:
        fig, ax = plt.subplots(figsize=(5, 5))

        for i, T in enumerate(temperatures):
            color = base_colors[i % len(base_colors)]
            plot_color = adjust_lightness(color, 0.7)
            
            # Soft 데이터 플로팅
            soft_path = os.path.join(base_path, anion, styles['soft'], f'1.0_{T}_pow.txt')
            if os.path.exists(soft_path):
                data_soft = np.loadtxt(soft_path, delimiter=',')
                ax.plot(data_soft[:, 0], data_soft[:, 1], label=f'{T} K, Soft', color=plot_color, linewidth=1.5)
            else:
                print(f"Warning: Soft data file not found for {anion} at {T}K: {soft_path}")

            # Hard 데이터 플로팅
            hard_path = os.path.join(base_path, anion, styles['hard'], f'1.0_{T}_pow.txt')
            if os.path.exists(hard_path):
                data_hard = np.loadtxt(hard_path, delimiter=',')
                ax.plot(data_hard[:, 0], data_hard[:, 1], label=f'{T} K, Hard', color=plot_color, linewidth=2.0, linestyle='dotted')
            else:
                print(f"Warning: Hard data file not found for {anion} at {T}K: {hard_path}")

        # 축 스케일 및 범위 설정
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e0, 1e5)
        ax.set_ylim(1e-6, 1e0)

        # 축 레이블 설정
        ax.set_xlabel('n(ps)', fontsize=14)
        ax.set_ylabel(r'$f(n)$', fontsize=14)

        # 범례 추가
        if anion == "fsi":
            ax.legend()

        # 플롯 내부에 텍스트 추가

        ax.set_title(anion.upper(), fontsize=14)
        # PDF 파일로 저장
        output_path = os.path.join(output_dir, f'{anion}_plot.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    create_plots()
