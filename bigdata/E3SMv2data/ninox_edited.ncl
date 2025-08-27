
; COMPUTE E3SMv2 NINO1+2, 3, 4, 3.4:
;  (1) Define the lat/lon box
;  (2) Take area weighted average
;  (3) Subtract the climatology (NCL)
;  (4) Do 5-month running mean  (NCL)
;  (6) Divide by standard deviation  (NCL)
; Contact: Po-Lun.Ma@pnnl.gov

begin

 mx   = "0151"
 ; ddir = "/pscratch/sd/q/qinyi/E3SMv2_init/v2.LR.historical_"+mx+"/archive/atm/hist"
 ddir = "/pscratch/sd/p/plutzner/E3SM/E3SMv2data/member"+mx+"/monthly_bilinear"
 edir = "/pscratch/sd/p/plma/shared/for_jolan"
 latS = (/-5., -10, -5, -5/)
 latN = (/5., 0, 5, 5/)
 lonL = (/190., 270, 210, 160/)
 lonR = (/240., 280, 270, 210/)

 fs=systemfunc ("ls "+ddir+"/v2.LR.*.eam.h0.*.nc")
 fa=addfiles (fs,"r")
 ListSetType (fa, "cat") 
 ts=fa[:]->TS
 dat=fa[:]->date
 ts@_FillValue = 9e36
 ListSetType (fa, "join") 
 lat=doubletofloat(fa[:]->lat)
 lon=doubletofloat(fa[:]->lon)
 f1=addfile(fs(0),"r")
 wgt1=f1->area
 
 print("set knowns")

 dim1=dimsizes(ts)
 tmp1 = new((/4,dim1(0),dim1(1)/),float)
 tmp1(0,:,:)=ts
 
 print("about to run lat lon box loop")

 do i=0, 3
  tmp1(i,:,:)=(/where(lat.ge.latS(i).and.lat.le.latN(i).and.lon.ge.lonL(i).and.lon.le.lonR(i),ts,ts@_FillValue)/)
 end do

 print("selected lat lon box")

 ninox = dim_avg_wgt_n_Wrap (tmp1,wgt1,1,2)
 
 do i=0,3
  ninox(i,:)=ninox(i,:)-avg(ninox(i,:))
  ninox(i,:)=runave_Wrap(ninox(i,:), 5, 0)
  ninox(i,:)=ninox(i,:)/stddev(ninox(i,:))
 end do

 system("rm -f "+edir+"/nino.member"+mx+".nc")
 fout=addfile(""+edir+"/nino.member"+mx+".nc","c")

 fout->date   = dat
 fout->nino34 = ninox(0,:)
 fout->nino12 = ninox(1,:)
 fout->nino3  = ninox(2,:)
 fout->nino4  = ninox(3,:)


end
