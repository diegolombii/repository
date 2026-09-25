"""
Bridge USB CDC PC <-> STM32. Protocollo:
    MCU -> PC : "REQ <k>"              chiede il chunk k
    PC  -> MCU: <CHUNK_SIZE> byte      il chunk k, letto da dataset_int8_ch<N>.bin
    MCU -> PC : "RES <k> <score> <label>"
    MCU -> PC : "DONE"

Il chunk k e' il file k-esimo generato da input.py; la label vera e' il
byte k-esimo di labels_uint8_ch<N>.bin.

Uso: python host_usb_bridge.py --port COM3 --channels 6
"""

import argparse
import csv
import os
import sys
import time
import serial

ROWS_PER_FILE = 7800
NUM_CHUNKS = 1500
OUTPUT_DIR = "risultati"


def parse_args():
    p = argparse.ArgumentParser(description="Bridge USB CDC PC <-> STM32")
    p.add_argument("--port", required=True, help="Porta seriale (es. COM3)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--channels", type=int, default=6, help="Deve combaciare con N_CHANNELS in main.c")
    p.add_argument("--dataset", default=None, help="default: dataset_int8_ch<N>.bin")
    p.add_argument("--true-labels", default=None, help="default: labels_uint8_ch<N>.bin")
    p.add_argument("--out-result", default=None, help="default: risultati/output_ch<N>.csv")
    p.add_argument("--out-report", default=None, help="default: risultati/output_ch<N>.txt")
    p.add_argument("--timeout", type=float, default=10.0)
    args = p.parse_args()

    n = args.channels
    defaults = {
        "dataset": f"dataset_int8_ch{n}.bin",
        "true_labels": f"labels_uint8_ch{n}.bin",
        "out_result": os.path.join(OUTPUT_DIR, f"output_ch{n}.csv"),
        "out_report": os.path.join(OUTPUT_DIR, f"output_ch{n}.txt"),
    }
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    return args


def open_serial_with_retry(port, baud, timeout, retry_delay=0.5):
    """Riprova ad aprire la porta finche' non ci riesce -- il reset della
    board causa una vera disconnessione/riconnessione USB."""
    warned = False
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=timeout)
            if warned:
                print(f"Porta {port} tornata disponibile, riaperta.")
            return ser
        except serial.SerialException:
            if not warned:
                print(f"Porta {port} non disponibile, riprovo ogni {retry_delay}s (Ctrl+C per interrompere)...")
                warned = True
            time.sleep(retry_delay)


def reconnect(ser, port, baud, timeout):
    print("Porta persa (probabile reset/disconnessione della board). Riconnessione...")
    try:
        ser.close()
    except Exception:
        pass
    return open_serial_with_retry(port, baud, timeout)


def main():
    args = parse_args()
    chunk_size = ROWS_PER_FILE * args.channels
    print(f"Canali: {args.channels} -> CHUNK_SIZE = {chunk_size} byte/chunk")
    print(f"Dataset: {args.dataset} | Label vere: {args.true_labels}")

    try:
        dataset = open(args.dataset, "rb")
        true_labels = open(args.true_labels, "rb").read()
    except OSError as ex:
        print(f"Impossibile aprire i file di input: {ex}")
        sys.exit(1)
    print(f"Label vere caricate: {len(true_labels)} valori")

    os.makedirs(os.path.dirname(args.out_result) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
    f_result = open(args.out_result, "w", newline="", encoding="utf-8")
    f_report = open(args.out_report, "w", encoding="utf-8")
    csv_writer = csv.writer(f_result)
    csv_writer.writerow(["chunk", "score", "label_predetta", "label_vera", "esito"])

    ser = open_serial_with_retry(args.port, args.baud, args.timeout)
    print(f"Porta {args.port} aperta. In attesa di dati dalla board...")

    chunks_done = corretti = 0
    t_start = time.time()

    try:
        while True:
            try:
                raw_line = ser.readline()
            except (serial.SerialException, OSError):
                ser = reconnect(ser, args.port, args.baud, args.timeout)
                continue

            if not raw_line:
                print("Timeout: nessuna riga ricevuta dalla board. Continuo ad aspettare...")
                continue

            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            parts = line.split()

            if parts[0] == "REQ":
                k = int(parts[1])
                dataset.seek(k * chunk_size)
                data = dataset.read(chunk_size)
                print(f"[PC] REQ {k}, invio {len(data)} byte")
                if len(data) != chunk_size:
                    print(f"ATTENZIONE: chunk {k} incompleto ({len(data)}/{chunk_size} byte) "
                          f"- controlla dimensione dataset o --channels.")
                try:
                    ser.write(data)
                except (serial.SerialException, OSError):
                    ser = reconnect(ser, args.port, args.baud, args.timeout)

            elif parts[0] == "RES":
                if len(parts) < 4:
                    print(f"ATTENZIONE: riga RES malformata, la ignoro: {line!r}")
                    continue
                try:
                    k, score, label_pred = int(parts[1]), float(parts[2]), int(parts[3])
                except ValueError:
                    print(f"ATTENZIONE: riga RES non parsabile, la ignoro: {line!r}")
                    continue

                if k < len(true_labels):
                    label_vera = true_labels[k]
                    esito = "OK" if label_pred == label_vera else "ERRORE"
                    corretti += esito == "OK"
                else:
                    label_vera, esito = "N/A", "N/A"
                    print(f"ATTENZIONE: nessuna label vera per il chunk {k}")

                csv_writer.writerow([k, f"{score:.6f}", label_pred, label_vera, esito])
                f_result.flush()
                chunks_done += 1

                if chunks_done % 50 == 0 or chunks_done == NUM_CHUNKS:
                    elapsed = time.time() - t_start
                    print(f"[{chunks_done}/{NUM_CHUNKS}] chunk {k}: score={score:.6f} "
                          f"pred={label_pred} vera={label_vera} {esito}  "
                          f"(accuracy: {corretti/chunks_done*100:.1f}%, t={elapsed:.1f}s)")

            elif parts[0] == "DONE":
                elapsed = time.time() - t_start
                accuracy = (corretti / chunks_done * 100) if chunks_done else 0.0
                print(f"Completato: {chunks_done}/{NUM_CHUNKS} chunk in {elapsed:.1f}s.")
                print(f"Accuracy finale: {corretti}/{chunks_done} ({accuracy:.2f}%)")
                break

            else:
                print(f"[BOARD] {line}")
                f_report.write(line + "\n")
                f_report.flush()

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
    finally:
        dataset.close()
        f_result.close()
        f_report.close()
        ser.close()
        print("File chiusi, porta seriale chiusa.")


if __name__ == "__main__":
    main()