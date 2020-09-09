# ssh nao@pepper #10.42.0.49
# Gets pepper up even if something is wrong (red led flashing)
# qicli call ALMotion.setDiagnosisEffectEnabled 0
# Pepper will now gladly run into walls, and move even if
# the charging port is open. Uncomment at your own risk.
qicli call ALMotion.setExternalCollisionProtectionEnabled 'All' 0
# Stops the head from moving autonomously (kinda)
qicli call ALBasicAwareness.pauseAwareness
