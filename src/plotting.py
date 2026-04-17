import matplotlib.pyplot as plt
import numpy as np

from .pf import pf_run
from .ipso_pf import ipso_pf_run
from .rul import predict_future


battery_names = ['B5', 'B6', 'B7', 'B18']

START_K = 80
PRED_HORIZON = 120
START_K2 = 78

def plot_capacity(results):
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.suptitle("Capacity Estimation: PF vs IPSO-PF (NASA Batteries)", fontsize=13, y=1.00)
    
    battery_names = ['B5', 'B6', 'B7', 'B18']
    colors = {'actual': 'purple', 'pf': 'orange', 'ipso': 'steelblue',
              'err_pf': 'orange', 'err_ipso': 'steelblue'}
    
    for row, name in enumerate(battery_names):
        r = results[name]
        y = r['y']
        cycles = np.arange(len(y))
    
        # --- Left: capacity curves ---
        ax_l = axes[row, 0]
        ax_l.plot(cycles, y,          color=colors['actual'], lw=1.2, label='Actual Capacity')
        ax_l.plot(cycles, r['cap_pf'],  color=colors['pf'],    lw=1.0, label='PF',      alpha=0.85)
        ax_l.plot(cycles, r['cap_ipso'], color=colors['ipso'], lw=1.0, label='IPSO-PF', alpha=0.85)
    
        ax_l.set_ylabel('Capacity (Ah)')
        ax_l.set_xlabel('Cycle Number')
        ax_l.set_title(f'{name}  (MAE: PF={r["mae_pf"]:.4f}, IPSO-PF={r["mae_ipo"]:.4f})')
        ax_l.legend(fontsize=7, loc='upper right')
        ax_l.text(0.02, 0.05, f'({chr(97+row*2)}-1)', transform=ax_l.transAxes, fontsize=8)
    
        # Inset zoom around mid-cycles
        mid = len(y) // 2
        axins = ax_l.inset_axes([0.55, 0.55, 0.40, 0.38])
        sl = slice(max(0, mid-5), mid+5)
        axins.plot(cycles[sl], y[sl],               color=colors['actual'], lw=1.2)
        axins.plot(cycles[sl], r['cap_pf'][sl],      color=colors['pf'],    lw=1.0)
        axins.plot(cycles[sl], r['cap_ipso'][sl],    color=colors['ipso'],  lw=1.0)
        axins.tick_params(labelsize=6)
    
        # --- Right: error curves ---
        ax_r = axes[row, 1]
        ax_r.plot(cycles, r['err_pf'],   color=colors['err_pf'],   lw=0.8, label='PF',      alpha=0.85)
        ax_r.plot(cycles, r['err_ipso'], color=colors['err_ipso'], lw=0.8, label='IPSO-PF', alpha=0.85)
        ax_r.axhline(0, color='k', lw=0.6, ls='--')
        ax_r.set_ylabel('Capacity Error (Ah)')
        ax_r.set_xlabel('Cycle Number')
        ax_r.set_title(f'{name} — Estimation Error')
        ax_r.legend(fontsize=7, loc='upper right')
        ax_r.text(0.02, 0.05, f'({chr(97+row*2+1)}-2)', transform=ax_r.transAxes, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('outputs/fig_capacity_estimation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_capacity_estimation.png")

def plot_rul_k58(results):
    START_K = 80      # prediction starting cycle
    PRED_HORIZON = 120  # predict up to this many cycles ahead
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
    fig2.suptitle(f"RUL Prediction Starting at Cycle K={START_K} (NASA Batteries)",
                  fontsize=13, y=1.00)
    
    for idx, name in enumerate(battery_names):
        ax = axes2[idx // 2, idx % 2]
        r = results[name]
        y = r['y']
        threshold = r['threshold']
        cycles = np.arange(len(y))
    
        # Re-run filters up to START_K to get particle distribution at that point
        y_train = y[:START_K]
        init_p  = r['init_p']
    
        _, _, xp_pf_k,  wp_pf_k  = pf_run(y_train, init_p)
        _, _, xp_ipo_k, wp_ipo_k = ipso_pf_run(y_train, init_p)
    
        # Predict future capacity from START_K
        pred_pf,   rul_pf   = predict_future(xp_pf_k,  wp_pf_k,  START_K, PRED_HORIZON, threshold)
        pred_ipo,  rul_ipo  = predict_future(xp_ipo_k, wp_ipo_k, START_K, PRED_HORIZON, threshold)
    
        future_cycles_pf  = np.arange(START_K, START_K + len(pred_pf))
        future_cycles_ipo = np.arange(START_K, START_K + len(pred_ipo))
    
        # Find actual RUL
        actual_rul = None
        for j, cap in enumerate(y):
            if cap <= threshold:
                actual_rul = j - START_K
                break
    
        # Plot
        ax.plot(cycles, y, color='steelblue', lw=1.2, label='Actual Capacity')
        ax.plot(future_cycles_pf,  pred_pf,  color='purple', lw=1.2, ls='-', label='PF Prediction')
        ax.plot(future_cycles_ipo, pred_ipo, color='orange',  lw=1.2, ls='-', label='IPSO-PF Prediction')
        ax.axhline(threshold, color='k', lw=1.0, label=f'Failure threshold ({threshold}Ah)')
        ax.axvline(START_K, color='green', lw=1.0, ls='--', label=f'Prediction Start (K={START_K})')
    
        # Mark predicted RUL endpoints
        if rul_pf < PRED_HORIZON:
            ax.axvline(START_K + rul_pf,  color='purple', lw=0.8, ls=':')
        if rul_ipo < PRED_HORIZON:
            ax.axvline(START_K + rul_ipo, color='orange',  lw=0.8, ls=':')
        if actual_rul is not None:
            ax.axvline(START_K + actual_rul, color='steelblue', lw=0.8, ls=':')
    
        # Annotate RUL values
        info  = f"PF RUL: {rul_pf}\nIPSO-PF RUL: {rul_ipo}"
        if actual_rul:
            info += f"\nActual RUL: {actual_rul}"
            ae_pf  = abs(rul_pf  - actual_rul)
            ae_ipo = abs(rul_ipo - actual_rul)
            info += f"\nAE PF: {ae_pf}  IPSO-PF: {ae_ipo}"
        ax.text(0.02, 0.08, info, transform=ax.transAxes,
                fontsize=7, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(f'RUL Prediction for {name}')
        ax.legend(fontsize=7, loc='upper right')
    
        # Inset zoom near failure threshold
        end_cycle = min(len(y) - 1, START_K + rul_ipo + 15)
        start_zoom = max(START_K + rul_ipo - 20, START_K)
        sl = slice(start_zoom, min(end_cycle + 5, len(y)))
        axins2 = ax.inset_axes([0.55, 0.35, 0.40, 0.35])
        axins2.plot(cycles[sl], y[sl], color='steelblue', lw=1.0)
        f_sl_pf  = [c for c in future_cycles_pf  if start_zoom <= c <= end_cycle + 5]
        f_sl_ipo = [c for c in future_cycles_ipo if start_zoom <= c <= end_cycle + 5]
        idx_pf  = [i for i, c in enumerate(future_cycles_pf)  if start_zoom <= c <= end_cycle + 5]
        idx_ipo = [i for i, c in enumerate(future_cycles_ipo) if start_zoom <= c <= end_cycle + 5]
        if idx_pf:  axins2.plot(f_sl_pf,  pred_pf[idx_pf],  color='purple', lw=1.0, ls='--')
        if idx_ipo: axins2.plot(f_sl_ipo, pred_ipo[idx_ipo], color='orange', lw=1.0, ls='--')
        axins2.axhline(threshold, color='k', lw=0.8)
        axins2.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.savefig('outputs/fig_rul_prediction_k58.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_rul_prediction_k58.png")

def plot_rul_k78(results):
    START_K2 = 78

    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
    fig3.suptitle(f"RUL Prediction Starting at Cycle K={START_K2} (NASA Batteries)",
                  fontsize=13, y=1.00)
    
    for idx, name in enumerate(battery_names):
        ax = axes3[idx // 2, idx % 2]
        r = results[name]
        y = r['y']
        threshold = r['threshold']
        cycles = np.arange(len(y))
    
        if START_K2 >= len(y):
            ax.text(0.5, 0.5, f'Not enough cycles for K={START_K2}',
                    ha='center', va='center', transform=ax.transAxes)
            continue
    
        y_train = y[:START_K2]
        init_p  = r['init_p']
    
        _, _, xp_pf_k,  wp_pf_k  = pf_run(y_train, init_p)
        _, _, xp_ipo_k, wp_ipo_k = ipso_pf_run(y_train, init_p)
    
        pred_pf,  rul_pf  = predict_future(xp_pf_k,  wp_pf_k,  START_K2, PRED_HORIZON, threshold)
        pred_ipo, rul_ipo = predict_future(xp_ipo_k, wp_ipo_k, START_K2, PRED_HORIZON, threshold)
    
        future_cycles_pf  = np.arange(START_K2, START_K2 + len(pred_pf))
        future_cycles_ipo = np.arange(START_K2, START_K2 + len(pred_ipo))
    
        actual_rul = None
        for j, cap in enumerate(y):
            if cap <= threshold:
                actual_rul = j - START_K2
                break
    
        ax.plot(cycles, y, color='steelblue', lw=1.2, label='Actual Capacity')
        ax.plot(future_cycles_pf,  pred_pf,  color='purple', lw=1.2, ls='--', label='PF Prediction')
        ax.plot(future_cycles_ipo, pred_ipo, color='orange',  lw=1.2, ls='--', label='IPSO-PF Prediction')
        ax.axhline(threshold, color='k', lw=1.0, label=f'Failure threshold ({threshold}Ah)')
        ax.axvline(START_K2, color='green', lw=1.0, ls='--', label=f'Prediction Start (K={START_K2})')
    
        if rul_pf  < PRED_HORIZON: ax.axvline(START_K2 + rul_pf,  color='purple', lw=0.8, ls=':')
        if rul_ipo < PRED_HORIZON: ax.axvline(START_K2 + rul_ipo, color='orange',  lw=0.8, ls=':')
        if actual_rul is not None: ax.axvline(START_K2 + actual_rul, color='steelblue', lw=0.8, ls=':')
    
        info = f"PF RUL: {rul_pf}\nIPSO-PF RUL: {rul_ipo}"
        if actual_rul:
            info += f"\nActual RUL: {actual_rul}"
        ax.text(0.02, 0.08, info, transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(f'RUL Prediction for {name}')
        ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('outputs/fig_rul_prediction_k78.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_rul_prediction_k78.png")

    
    
    
    
    
    
    