open Altaz

let (%.) = mod_float

(* Inverse of raDectoAltAz - converts altitude and azimuth to RA and Dec *)
let altAztoRaDec _Alt _Az _Lat _Long _LST =
  (* Convert degrees to radians *)
  let alt_rad = _Alt *. (Float.pi /. 180.) in
  let az_rad = _Az *. (Float.pi /. 180.) in
  let lat_rad = _Lat *. (Float.pi /. 180.) in
  
  (* Convert horizontal coordinates to equatorial *)
  let sin_dec = sin(alt_rad) *. sin(lat_rad) +. cos(alt_rad) *. cos(lat_rad) *. cos(az_rad) in
  let dec = asin(sin_dec) *. (180. /. Float.pi) in
  
  (* Calculate hour angle *)
  let cos_h = (sin(alt_rad) -. sin(lat_rad) *. sin_dec) /. (cos(lat_rad) *. cos(dec *. (Float.pi /. 180.))) in
  let sin_h = -1. *. cos(alt_rad) *. sin(az_rad) /. cos(dec *. (Float.pi /. 180.)) in
  let ha_rad = atan2 sin_h cos_h in
  let ha_hours = ha_rad *. (12. /. Float.pi) in
  
  (* Ensure HA is in proper range (-12 to +12 hours) *)
  let ha_hours = if ha_hours < -12. then ha_hours +. 24. else if ha_hours > 12. then ha_hours -. 24. else ha_hours in
  
  (* Calculate RA from LST and hour angle *)
  let ra_hours = _LST -. ha_hours in
  (* Ensure RA is in range 0-24 hours *)
  let ra_hours = (ra_hours +. 24.) %. 24. in
  
  (* Convert RA from hours to degrees *)
  let ra_deg = ra_hours *. 15. in
  
  (* Return RA, Dec, and hour angle *)
  ra_deg, dec, ha_hours

(* Function to convert from Alt/Az to J2000 coordinates *)
let altaz_to_j2000 alt az latitude longitude =
  (* Get current time *)
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  let yr = tm.tm_year + 1900 in
  let mon = tm.tm_mon + 1 in
  let dy = tm.tm_mday in
  let hr = tm.tm_hour in
  let min = tm.tm_min in
  let sec = tm.tm_sec in
  
  (* Calculate Julian date *)
  let jd_calc = computeTheJulianDay true yr mon dy +. float_of_int(hr*3600+min*60+sec) /. 86400.0 in
  
  (* Calculate local sidereal time *)
  let lst = local_siderial_time' longitude (jd_calc -. jd_2000) in
  
  (* Convert Alt/Az to current epoch RA/Dec *)
  let ra_now, dec_now, _ = altAztoRaDec alt az latitude longitude lst in
  
  (* Now convert current epoch to J2000 (inverse of j2000_to_jnow) *)
  let date = Unix.gettimeofday() in
  let datum,_ = Unix.mktime {tm_sec=0; tm_min=0; tm_hour=12; tm_mday=1; tm_mon=0; tm_year = 100; tm_wday=0; tm_yday=0; tm_isdst=false} in
  let _T = (date -. datum) /. 86400.0 /. 36525.0 in
  let _M = 1.2812323 *. _T +. 0.0003879 *. _T *. _T +. 0.0000101 *. _T *. _T *. _T in
  let _N = 0.5567530 *. _T -. 0.0001185 *. _T *. _T +. 0.0000116 *. _T *. _T *. _T in
  
  (* Apply the inverse correction to get J2000 coordinates *)
  let delta_ra = _M +. _N *. sin (ra_now *. (Float.pi /. 180.)) *. tan (dec_now *. (Float.pi /. 180.)) in
  let delta_dec = _N *. cos (ra_now *. (Float.pi /. 180.)) in
  let ra2000 = ra_now -. delta_ra in
  let dec2000 = dec_now -. delta_dec in
  
  ra2000, dec2000

(* Main function to calculate Alt/Az from J2000 coordinates at specified time *)
let altaz_to_j2000_time yr mon dy hr min sec alt az latitude longitude =
  (* Calculate Julian date *)
  let jd_calc = computeTheJulianDay true yr mon dy +. float_of_int(hr*3600+min*60+sec) /. 86400.0 in
  
  (* Calculate local sidereal time *)
  let lst = local_siderial_time' longitude (jd_calc -. jd_2000) in
  
  (* Convert Alt/Az to current epoch RA/Dec *)
  let ra_now, dec_now, ha_now = altAztoRaDec alt az latitude longitude lst in
  
  (* Calculate time offset in Julian centuries *)
  let datum,_ = Unix.mktime {tm_sec=0; tm_min=0; tm_hour=12; tm_mday=1; tm_mon=0; tm_year = 100; tm_wday=0; tm_yday=0; tm_isdst=false} in
  let _T = (jd_calc -. jd_2000) /. 36525.0 in
  
  (* Calculate correction factors *)
  let _M = 1.2812323 *. _T +. 0.0003879 *. _T *. _T +. 0.0000101 *. _T *. _T *. _T in
  let _N = 0.5567530 *. _T -. 0.0001185 *. _T *. _T +. 0.0000116 *. _T *. _T *. _T in
  
  (* Apply the inverse correction to get J2000 coordinates *)
  let delta_ra = _M +. _N *. sin (ra_now *. (Float.pi /. 180.)) *. tan (dec_now *. (Float.pi /. 180.)) in
  let delta_dec = _N *. cos (ra_now *. (Float.pi /. 180.)) in
  let ra2000 = ra_now -. delta_ra in
  let dec2000 = dec_now -. delta_dec in
  
  jd_calc, ra2000, dec2000, ra_now, dec_now, lst, ha_now

(* Helper function to test the conversion - converts from RA/Dec to Alt/Az and back *)
let test_conversion ra dec lat long =
  let date = Unix.gmtime (Unix.gettimeofday ()) in
  let yr = date.tm_year + 1900 in
  let mon = date.tm_mon + 1 in
  let dy = date.tm_mday in
  let hr = date.tm_hour in
  let min = date.tm_min in
  let sec = date.tm_sec in
  
  Printf.printf "Original RA: %f Dec: %f\n" ra dec;
  
  (* Convert from RA/Dec to Alt/Az *)
  let jd, ra_now, dec_now, alt, az, lst, ha = altaz_calc yr mon dy hr min sec ra dec lat long in
  Printf.printf "Alt: %f Az: %f LST: %f\n" alt az lst;
  
  (* Convert back from Alt/Az to RA/Dec *)
  let ra_calc, dec_calc, ha_calc = altAztoRaDec alt az lat long lst in
  Printf.printf "Calculated RA: %f Dec: %f\n" ra_calc dec_calc;
  Printf.printf "Difference RA: %f Dec: %f\n" (ra_now -. ra_calc) (dec_now -. dec_calc);
  
  (* Return the results *)
  jd, alt, az, lst, ha, ra_calc, dec_calc, ha_calc
