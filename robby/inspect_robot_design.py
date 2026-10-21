"""Stream-inspect user-supplied design files without extracting a large mesh.

Outputs metadata/hashes only; original images, PDFs and mesh licenses are not
assumed transferable, and this does not publish them. The procedural renderer
uses the proposal's nominal dimensions rather than copying the dense mesh.
"""
import argparse,hashlib,json,xml.sax,zipfile
from pathlib import Path

class Geometry(xml.sax.ContentHandler):
    def __init__(self):
        super().__init__();self.vertices=0;self.triangles=0;self.bounds=[[float('inf')]*3,[float('-inf')]*3];self.unit=None
    def startElement(self,name,attrs):
        if name=='model':self.unit=attrs.get('unit','millimeter')
        if name=='triangle':self.triangles+=1
        if name=='vertex':
            self.vertices+=1
            for i,key in enumerate('xyz'):
                value=float(attrs[key]);self.bounds[0][i]=min(self.bounds[0][i],value);self.bounds[1][i]=max(self.bounds[1][i],value)

def inspect(root):
    names=['Roboterkopf_Displaymontage.pdf','Roboterkopf_Explosionsansicht.svg',
        'Mini_Roboterkopf_Prototyp_Bauplan_Waveshare.pdf','Roboterkopf_Masszeichnung.svg',
        'cute+robot+3d+model.3mf','cute+robot+3d+model.zip','desk-robi.png']
    rows=[]
    for name in names:
        path=root/name;digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        rows.append({'name':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
    geometry=Geometry();parser=xml.sax.make_parser();parser.setFeature(xml.sax.handler.feature_external_ges,False)
    parser.setContentHandler(geometry)
    with zipfile.ZipFile(root/'cute+robot+3d+model.3mf') as archive:
        entry=archive.getinfo('3D/3dmodel.model')
        if entry.file_size>256*1024*1024:raise ValueError('Mesh XML exceeds bounded inspection budget')
        with archive.open(entry) as stream:parser.parse(stream)
    return {'files':rows,'original_3mf':{'vertices':geometry.vertices,'triangles':geometry.triangles,
        'unit':geometry.unit,'raw_vertex_bounds_not_assembled_CAD':geometry.bounds},
        'migration_json':'Cache-migration audit, not a robot design file; excluded from geometry.',
        'source_files_extracted':False,'source_files_published':False,
        'renderer':'Original compact procedural reconstruction; nominal design dimensions, not manufacturing CAD.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('/mnt/nvme/robo-speaker-studies'))
    result=inspect(p.parse_args().directory)
    from speaker_v4.common import atomic
    atomic(Path(__file__).parent/'design-inspection.json',result);print(json.dumps(result,indent=2))
