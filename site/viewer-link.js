export const playbackSpeed=Object.freeze({min:.25,max:1.5,default:1,unit:'×'});
export function normalizePlaybackSpeed(value){
  const speed=Number(value);
  return Number.isFinite(speed)&&speed>=playbackSpeed.min&&speed<=playbackSpeed.max?speed:playbackSpeed.default;
}
export function readTerrainLink(query){return {enabled:query.get('terrain')==='1',speed:normalizePlaybackSpeed(query.get('speed'))};}
export function writeTerrainLink(query,state){
  query.delete('terrain');query.delete('speed');
  if(state?.enabled){query.set('terrain','1');query.set('speed',String(normalizePlaybackSpeed(state.speed)));}
  return query;
}
