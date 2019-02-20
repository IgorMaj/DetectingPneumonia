# DetectingPneumonia
Projekat iz predmeta SOFT kompjuting. 

U pitanju je konvolutivna neuronska mreza, koja klasifikuje rendgenske snimke pluća na tri kategorije:
<br/><br/>
<b>- Bakterijsku pneumoniu</b><br/>
<b>- Virusnu pneumoniu</b><br/>
<b>- Normalan nalaz</b><br/>

Skup podataka je moguće pronaći <a href="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia">ovde</a>. Neophodno ga je otpakovati
i staviti u root direktorijuma. 
Program se može pokrenuti sa nekoliko argumenata komandne linije. Svi argumenti su opcioni i imaju podrazumevane vrednosti.
<br/>
<br/>
`--load_model_from_file=False ili --load_model_from_file=True`<br/><b>Određuje da li da se model učita iz fajla `model.dat`.<br/>
Default: False</b><br/><br/>
`--num_epochs=num`<br/><b>Specifiše broj epoha. Default: 5</b><br/><br/>
`--run_test=False ili --run_test=True`<br/><b>Određuje da li da se postojeći model evaluira na test skupu. Default: False</b><br/><br/>
`--runs_on_cloud=False ili --runs_on_cloud=True`<br/><b>Prilagođava način pokretanja Googlovom ml enginu,<br/>koristiti samo ako se pokreće u tom okruženju. Default: False</b><br/><br/>



