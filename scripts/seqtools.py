from IPython import embed
import pandas as pd



def dl_genome(id, folder='genomes'): # be sure get CORRECT id... 
    from glob import glob
    files =glob('%s/*.gb'%folder)
    out_file = '%s\\%s.gb'%(folder, id)
    
    if out_file in files:
        print out_file, 'already downloaded'
        return
    else:
        print 'downloading %s from NCBI'%id
    from Bio import Entrez
    Entrez.email = "jmonk@ucsd.edu" 
    handle = Entrez.efetch(db="nucleotide", id=id, rettype="gb", retmode="text")
    fout = open(out_file,'w')
    fout.write(handle.read())
    fout.close()

    
def make_blast_db(id,db_type='prot', overwrite=False):
    import os
    if os.path.isfile(id+'.nhr') and overwrite==False:
        print id, db_type,' blast db exists, skipping'
        return
    # out_file ='%s\\%s.fa.pin'%(folder, id)
    # from glob import glob
    # files =glob('%s/*.fa.pin'%folder)
    # if out_file in files:
        # print id, 'already has a blast db'
        # return
    
    """ make a blast db (BLAST 2.2.26+) """
    cmd_line='makeblastdb -in %s -dbtype %s' %(id,db_type)
    print 'making blast db with following command line...'
    print cmd_line
    os.system(cmd_line)

def run_blastp(seq,db,in_folder='prots', out_folder='bbh', \
    out=None,outfmt=6,evalue=0.001,threads=1):
    import os
    if out==None:
        out='%s\\%s_vs_%s.txt'%(out_folder, seq, db)
        
    
    from glob import glob
    files =glob('%s/*.txt'%out_folder)
    if out in files:
        print seq, 'already blasted'
        return
    
    print 'blasting %s vs %s'%(seq, db)
    
    db = '%s/%s.fa'%(in_folder, db)
    seq = '%s/%s.fa'%(in_folder, seq)
    cmd_line='blastp -db %s -query %s -out %s -evalue %s -outfmt %s -num_threads %i' \
    %(db, seq, out, evalue, outfmt, threads)
    
    print 'running blastp with following command line...'
    print cmd_line
    os.system(cmd_line)
    return out
    
def run_blastn(seq,db,out=None,outfmt=6,evalue=0.001,threads=1, strand='both'):
    import os
        
    print 'blasting %s vs %s'%(seq, db)
    
    db = '%s'%(db)
    seq = '%s'%(seq)
    cmd_line='blastn -db %s -query %s -out %s -evalue %s -outfmt %s -strand %s -num_threads %i' \
    %(db, seq, out, evalue, outfmt, strand, threads)
    
    print 'running blastn with following command line...'
    print cmd_line
    os.system(cmd_line)
    return out
    
def get_gene_lens(query, in_folder='prots'):
    from Bio import SeqIO
    file = '%s/%s.fa'%(in_folder, query)
    handle = open(file)
    records = SeqIO.parse(handle, "fasta")
    out = []
    for record in records:
        out.append({'gene':record.name, 'gene_length':len(record.seq)})
    
    out = pd.DataFrame(out)
    return out
    


def convert_to_fasta(file):
    outfile = file[:-2]+'fasta'
    fout = open(outfile,'w')
    from Bio import SeqIO
    handle=open(file)
    out = []
    Ncount = 0
    for record in SeqIO.parse(handle,"genbank"):
        # print record.seq
        fout.write('>%s\n%s\n'%(record.id, record.seq))
        for s in record.seq:
            out.append(s)
            if s=='N':
                Ncount+=1
        out = list(set(out))
    
    fout.close()
    return outfile
    
def get_gene_seq(gbk_file, gene, upstream=0, downstream=0):
    from Bio import SeqIO
    handle = open(gbk_file)
    records = SeqIO.parse(handle, "genbank")
    for record in records:
        for f in record.features:
            if f.type=='gene' and 'gene' in f.qualifiers:
                if gene in f.qualifiers['gene']:
                    start = f.location.start
                    stop = f.location.end
                    # embed()
                    if f.strand==1:
                        start=start+upstream
                        stop=stop+downstream
                        seq = record[start:stop]
                        return seq.seq.tostring()
                    elif f.strand==-1:
                        stop=stop+upstream
                        start=start+downstream
                        seq = record[start:stop]
                        # embed()
                        return seq.reverse_complement().seq.tostring()
                        
def blast_genome_for_seq(g,gene):
    g = convert_to_fasta(g)
    make_blast_db(g,db_type='nucl')
    run_blastn('%s'%gene,g, out='temp.csv', evalue=0.01, threads=6)
    cols = ['gene', 'subject', 'PID', 'alnLength', 'mismatchCount', 'gapOpenCount', 'queryStart', 'queryEnd', 'subjectStart', 'subjectEnd', 'eVal', 'bitScore']
    data = pd.read_csv('temp.csv', names=cols, sep='\t')
    
    contig = data.loc[0,'subject']
    start = data.loc[0,'subjectStart']-1
    end = data.loc[0,'subjectEnd']
    pid = data.loc[0,'PID']
    print 'found %s'%gene, contig, start, end, pid
    
    seq = extract_seq(g, contig, start, end)
   
    return seq

    
def extract_seq(g, contig, start, end):
    from Bio import SeqIO
    handle = open(g)
    records = SeqIO.parse(handle, "fasta")
    
    for record in records:
        if record.name==contig:
            if end>start:
                section = record[start:end]
            else:
                section = record[end-1:start+1].reverse_complement()
                
            seq = str(section.seq)
    return seq

    
if __name__=='__main__':
    
    ''' Download a genome'''
    # id = 'AE014075' # E. coli CFT073
    # dl_genome(id, folder='genomes')
    
    '''convert it to fasta'''
    k12genome = 'genomes/k12mg1655.gb'
    # convert_to_fasta(k12genome)
    
    ''' get sequence (+up/downstream) in that genome '''
    # gene='pgi'
    gene='tpiA'
    seq = get_gene_seq(k12genome, gene, upstream=200,downstream=0)
    seq_file = gene+'.fasta'
    fout=open(seq_file,'w')
    fout.write('>%s\n%s\n'%(gene,seq))
    fout.close()
    
    ''' blast for that seq against another genome '''
    seq2 = blast_genome_for_seq('genomes/AE014075.gb',seq_file)
    embed()
    
    
    

